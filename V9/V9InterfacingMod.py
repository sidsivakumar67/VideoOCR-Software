"""V9InterfacingMod - orchestration: video intake, frame sampling, filtering,
per-frame reading, validation, and Excel output. The back-end is called per
frame; the validation post-pass runs after all frames are read. This module
owns no model logic itself - it composes the back-end and validation behind
their interfaces (HARD RULE #7).

The selected filter is applied at SAMPLE time: the image written to all_frames/
is exactly what the model reads, so every filter is auditable on disk. All
results for a video live under Outputs/<video_name>/ (resolved against the V9
project root, so the location does not depend on the launch directory).
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from openpyxl import Workbook

from V9ConfigMod import Config, FilterParams
from V9BackendMod import Engine, LMStudioEngine, ReadResult
from V9ValidationMod import FrameValidation, validate

# V9 project root = the Modules/ directory's parent. Output paths resolve against
# this so results land in V9/Outputs/ regardless of where the script is launched.
_V9_ROOT = Path(__file__).resolve().parent.parent


def _outputs_root(cfg: Config) -> Path:
    """Resolve the configured output root against the V9 project root."""
    root = Path(cfg.outputs_dir)
    return root if root.is_absolute() else _V9_ROOT / root


def _fmt_hms(seconds: float) -> str:
    """Humanize a DURATION as HH:MM:SS.mmm (milliseconds dropped at >= 1 hour).

    For runtime/elapsed display only - never used for the spreadsheet's per-frame
    video timestamps, which stay integer seconds.
    """
    s = max(0.0, float(seconds))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    if h >= 1:
        return f"{h:02d}:{m:02d}:{int(sec):02d}"  # ms negligible at hour scale
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


@dataclass
class SampledFrame:
    second: int        # 0-based time index (seconds from the first sample)
    frame_path: Path   # saved sampled frame (already filtered) in all_frames/


@dataclass
class PartialRun:
    """Snapshot of a cancelled run - enough to optionally write a partial Excel."""
    cfg: Config
    base: Path
    settings_path: Path
    frames: list[SampledFrame]
    results: list[ReadResult]
    judge_results: list[ReadResult] | None
    stage: str


class Cancelled(Exception):
    """Raised when the user cancels a run; carries a PartialRun for optional save."""
    def __init__(self, partial: PartialRun | None = None):
        super().__init__("Run cancelled by user")
        self.partial = partial


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------
def _adaptive_factor(h: int, w: int, p: FilterParams) -> float:
    """Scale factor: enlarge small frames toward target, clamp to the ceiling.

    Small crops are upscaled (capped) to give the reader more pixels; large
    frames are left alone; anything above the ceiling is clamped down so the
    encoded image never blows past the model context (which yields empty reads).
    """
    long = max(h, w)
    if long < p.upscale_target_long_side:
        factor = min(p.upscale_target_long_side / long, p.upscale_max_factor)
    else:
        factor = 1.0
    if long * factor > p.upscale_max_long_side:
        factor = p.upscale_max_long_side / long
    return factor


def _resize_adaptive(img: np.ndarray, p: FilterParams) -> np.ndarray:
    h, w = img.shape[:2]
    f = _adaptive_factor(h, w, p)
    if abs(f - 1.0) < 1e-3:
        return img
    interp = cv2.INTER_CUBIC if f > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, None, fx=f, fy=f, interpolation=interp)


def apply_filter(img: np.ndarray, filter_name: str, p: FilterParams) -> np.ndarray:
    """Apply the named filter to a BGR frame, returning a BGR image to save/read.

    'unfiltered' is the original frame untouched. Every other variant converts
    to grayscale, applies its transform, then adaptively upscales (capped).
    """
    if filter_name == "unfiltered":
        return img  # original frame, untouched (intentionally not size-capped)

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if filter_name == "greyscale":
        out = g
    elif filter_name == "contrast":
        clahe = cv2.createCLAHE(
            clipLimit=p.contrast_clip_limit,
            tileGridSize=(p.contrast_tile_grid, p.contrast_tile_grid),
        )
        out = clahe.apply(g)
    elif filter_name == "black_and_white":
        if p.binary_threshold is None:
            _, out = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, out = cv2.threshold(g, p.binary_threshold, 255, cv2.THRESH_BINARY)
    elif filter_name == "adaptive":
        method = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if p.adaptive_method == "gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        out = cv2.adaptiveThreshold(
            g, 255, method, cv2.THRESH_BINARY, p.adaptive_block_size, p.adaptive_c
        )
    else:
        raise ValueError(f"Unknown filter: {filter_name!r}")

    out_bgr = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return _resize_adaptive(out_bgr, p)


# --------------------------------------------------------------------------
# Intake
# --------------------------------------------------------------------------
def intake(video_path: str | Path, cfg: Config) -> tuple[Path, Path, Path]:
    """Verify the video exists and create Outputs/<name>/{all_frames,flagged_frames}.

    Returns (base_dir, frames_dir, flagged_dir). base_dir also holds the xlsx and
    settings.txt. Re-running the same video overwrites its folder.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    base = _outputs_root(cfg) / video_path.stem
    frames_dir = base / "all_frames"
    flagged_dir = base / "flagged_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    flagged_dir.mkdir(parents=True, exist_ok=True)
    return base, frames_dir, flagged_dir


# --------------------------------------------------------------------------
# Sampling (no AI) - the selected filter is applied here, at save time
# --------------------------------------------------------------------------
def sample_frames(
    video_path: str | Path, cfg: Config, frames_dir: Path,
    progress=None, cancel=None, sink=None,
) -> list[SampledFrame]:
    """Sample one frame per window (1/samples_per_second), filter it, and save it.

    laplacian (default): pick the sharpest frame in each window by variance of
    the Laplacian. first_frame: take the first frame of each window. The saved
    image is the SELECTED FILTER applied to the chosen frame - i.e. exactly what
    the model will read. `progress`, if given, is called as
    progress(stage, current, total). If a `cancel` Event is set, raise Cancelled
    at the next window boundary. `sink`, if given, is the list to append into (so
    a caller still holds the partial frames after a cancellation).
    """
    frames: list[SampledFrame] = sink if sink is not None else []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            raise RuntimeError(f"Invalid FPS ({fps}) for video: {video_path}")
        window = max(1, int(round(fps / cfg.samples_per_second)))  # frames per sample
        est_total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // window)
        method = cfg.sampling_method
        second = 0
        # Walk the video one window at a time, keeping exactly one frame per window.
        # The final window may be short (partial trailing second) - still sampled.
        while True:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            best_sharp: float | None = None
            best_frame: np.ndarray | None = None
            got_any = False
            for k in range(window):
                ok, fr = cap.read()
                if not ok:
                    break
                got_any = True  # read every frame in the window so capture stays positioned
                if method == "first_frame":
                    if k == 0:
                        best_frame = fr
                elif method == "laplacian":
                    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if best_sharp is None or sharp > best_sharp:
                        best_sharp, best_frame = sharp, fr
                else:
                    raise ValueError(f"Unknown sampling method: {method!r}")
            if not got_any or best_frame is None:
                break
            # Apply the selected filter now: the file on disk IS the model input.
            filtered = apply_filter(best_frame, cfg.filter_name, cfg.filter_params)
            frame_path = frames_dir / f"t{second:04d}.png"
            cv2.imwrite(str(frame_path), filtered)
            frames.append(SampledFrame(second=second, frame_path=frame_path))
            second += 1
            if progress is not None:
                progress("sampling", second, est_total)
        return frames
    finally:
        cap.release()


# --------------------------------------------------------------------------
# Reading (per frame, stateless via the engine)
# --------------------------------------------------------------------------
def read_frames(
    frames: list[SampledFrame], engine: Engine,
    progress=None, stage: str = "reading", cancel=None, sink=None,
) -> list[ReadResult]:
    """Read each saved (already-filtered) frame through the engine.

    The frame on disk is already the filtered model input (see sample_frames), so
    no filtering happens here. `progress`, if given, is called as
    progress(stage, current, total). If a `cancel` Event is set, raise Cancelled
    at the next frame boundary. `sink`, if given, is the list to append into (so a
    caller still holds the partial reads after a cancellation).
    """
    results: list[ReadResult] = sink if sink is not None else []
    total = len(frames)
    for i, sf in enumerate(frames, start=1):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        # engine.read accepts a path and fails fast if the file is missing.
        results.append(engine.read(sf.frame_path))
        if progress is not None:
            progress(stage, i, total)
    return results


# --------------------------------------------------------------------------
# Excel output
# --------------------------------------------------------------------------
EXCEL_HEADERS = [
    "time_s", "frame", "raw_reading", "judge_reading", "parsed_value",
    "format_check", "range_check", "temporal_check", "judge", "flagged",
    "flag_reason", "confidence",
]


def write_excel(
    video_name: str,
    frames: list[SampledFrame],
    results: list[ReadResult],
    validations: list[FrameValidation],
    out_path: str | Path,
) -> Path:
    """Write one row per sampled second from the validation results.

    parsed_value is written as a real number (float) so the column charts cleanly
    and avoids Excel's "number stored as text" warning. raw_reading and
    judge_reading are kept as the models' verbatim text - never altered or coerced
    (HARD RULE #2). The check columns are empty when their check is off.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = video_name[:31]  # Excel sheet-name length limit
    ws.append(EXCEL_HEADERS)
    for sf, r, fv in zip(frames, results, validations):
        parsed_num = float(fv.parsed_value) if fv.parsed_value is not None else ""
        ws.append([
            sf.second,
            sf.frame_path.name,
            r.raw_text,                                        # verbatim back-end read
            fv.judge_raw if fv.judge_raw is not None else "",  # verbatim judge read
            parsed_num,                                        # numeric (float) or blank
            fv.format_check,
            fv.range_check,
            fv.temporal_check,
            fv.judge,
            "yes" if fv.flagged else "no",
            ";".join(fv.flag_reasons),
            "N/A" if fv.confidence is None else fv.confidence,
        ])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    return out_path


# --------------------------------------------------------------------------
# Hardware (which GPU is LM Studio actually using?)
# --------------------------------------------------------------------------
def _nvidia_smi_query() -> list[tuple[int, int, str]] | None:
    """One-shot nvidia-smi snapshot: [(mem_used_MiB, util_pct, name), ...] or None.

    Returns None when nvidia-smi is absent or errors (e.g. no NVIDIA driver).
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,utilization.gpu,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    rows: list[tuple[int, int, str]] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                rows.append((int(parts[0]), int(parts[1]), parts[2]))
            except ValueError:
                continue
    return rows or None


def check_gpu(cfg: Config) -> str:
    """Probe whether LM Studio runs the back-end model on the NVIDIA dGPU.

    Fires a few tiny reads at the configured back-end model while polling
    nvidia-smi GPU utilisation. A util spike during our probe means the dGPU is
    doing the work (green); a flat dGPU means the model is on the Intel iGPU/CPU
    (yellow). Returns a human-readable verdict for the GUI. It cannot name the
    Intel device specifically - "not the dGPU" is the actionable signal.
    """
    base = _nvidia_smi_query()
    if base is None:
        return "⚠️ nvidia-smi unavailable - cannot detect GPU (NVIDIA driver missing?)."
    gpu_name = base[0][2]

    engine = LMStudioEngine(
        cfg.base_url, cfg.backend_model_id, cfg.system_prompt,
        cfg.temperature, cfg.top_k,
    )
    try:
        engine.verify_model_available()
    except Exception as e:  # surface clearly, never crash the GUI
        return f"⚠️ Cannot reach LM Studio / back-end model: {e}"

    probe = np.full((64, 128, 3), 30, np.uint8)
    cv2.putText(probe, "8.8", (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    ok, buf = cv2.imencode(".png", probe)
    if not ok:
        return "⚠️ Could not build a probe image."
    img = buf.tobytes()

    util_peak = 0
    polling = True

    def _poll() -> None:
        nonlocal util_peak
        while polling:
            snap = _nvidia_smi_query()
            if snap:
                util_peak = max(util_peak, max(r[1] for r in snap))
            time.sleep(0.15)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    try:
        for _ in range(3):  # a few reads so the poller catches the util spike
            engine.read(img)
    except Exception as e:
        return f"⚠️ Probe read failed: {e}"
    finally:
        polling = False
        t.join(timeout=1)

    after = _nvidia_smi_query()
    mem_after = max(r[0] for r in after) if after else base[0][0]
    if util_peak >= 15:  # the dGPU clearly did our probe's compute
        return (f"🟢 dGPU in use - {gpu_name} "
                f"(VRAM {mem_after} MiB, peak util {util_peak}% during probe).")
    return (f"🟡 NOT on dGPU - {gpu_name} stayed idle during the probe "
            f"(peak util {util_peak}%). Model is likely on the Intel iGPU/CPU. "
            "Fix LM Studio GPU offload and re-check.")


# --------------------------------------------------------------------------
# Settings / run log
# --------------------------------------------------------------------------
def _write_settings_header(path: Path, video_path: Path, cfg: Config) -> None:
    """Write the settings used for this run; timing lines are appended later."""
    v = cfg.validation
    lines = [
        "Video OCR Software V9 - run settings",
        f"video: {video_path.name}",
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "[sampling]",
        f"samples_per_second: {cfg.samples_per_second}",
        f"sampling_method: {cfg.sampling_method}",
        "",
        "[filtering]",
        f"filter_name: {cfg.filter_name}",
        "",
        "[reading/format]",
        f"format_mask: {cfg.format_mask}",
        f"allow_negative: {cfg.allow_negative}",
        f"system_prompt: {cfg.system_prompt}",
        "",
        "[connection]",
        f"base_url: {cfg.base_url}",
        f"backend_model_id: {cfg.backend_model_id}",
        f"judge_model_id: {cfg.judge_model_id}",
        f"temperature: {cfg.temperature}",
        f"top_k: {cfg.top_k}",
        "",
        "[validation]",
        f"enabled: {v.enabled}",
        f"format_check: {v.format_check}",
        f"range_check: {v.range_check}",
        f"range_min: {v.range_min}",
        f"range_max: {v.range_max}",
        f"temporal_check: {v.temporal_check}",
        f"temporal_window: {v.temporal_window}",
        f"temporal_cutoff: {v.temporal_cutoff}",
        f"judge: {v.judge}",
        f"judge_swap_mode: {v.judge_swap_mode}",
        "",
        "[hardware]",
    ]
    # Passive snapshot (no probe inference) of what the NVIDIA GPU shows at start.
    smi = _nvidia_smi_query()
    if smi:
        for mem, util, name in smi:
            lines.append(f"nvidia_gpu: {name} (VRAM in use {mem} MiB, util {util}%)")
    else:
        lines.append("nvidia_gpu: nvidia-smi unavailable")
    lines += ["", "[timing]"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Partial finalize (after a cancellation, on user request)
# --------------------------------------------------------------------------
def finalize_partial(partial: PartialRun) -> Path:
    """Write a clearly-stamped PARTIAL Excel from a cancelled run's completed reads.

    Validates only the frames that were actually read (judge results aligned to
    whatever completed). The file is named <name>_PARTIAL.xlsx and both the sheet
    and settings.txt are stamped, so a partial is never mistaken for a full run.
    """
    cfg = partial.cfg
    results = partial.results
    frames = partial.frames[:len(results)]  # only frames actually read
    jr = partial.judge_results
    if jr is not None:
        m = min(len(results), len(jr), len(frames))
        results, jr, frames = results[:m], jr[:m], frames[:m]
    validations = validate(results, jr, cfg)

    flagged_dir = partial.base / "flagged_frames"
    flagged_dir.mkdir(parents=True, exist_ok=True)
    for sf, fv in zip(frames, validations):
        if fv.flagged:
            shutil.copy2(sf.frame_path, flagged_dir / sf.frame_path.name)

    stamp = f"CANCELLED - PARTIAL ({len(results)} of {len(partial.frames)} frames read)"
    with partial.settings_path.open("a", encoding="utf-8") as f:
        f.write(stamp + "\n")

    out_path = partial.base / f"{partial.base.name}_PARTIAL.xlsx"
    write_excel(f"{partial.base.name} (PARTIAL)", frames, results, validations, out_path)
    return out_path


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def process_video(
    video_path: str | Path,
    cfg: Config,
    engine: Engine | None = None,
    judge_engine: Engine | None = None,
    on_manual_swap=None,
    progress=None,
    on_flagged=None,
    log=None,
    cancel=None,
) -> Path:
    """Full pipeline: intake -> sample+filter -> read -> validate -> Excel.

    Each stage is wall-clock timed; timing lines (humanized HH:MM:SS.mmm) are
    appended to settings.txt and pushed to `log(line)` (the front-end console
    pane). The back-end pass completes fully, then (if the judge is enabled) the
    judge pass runs as an independent second read pass. Validation flags readings
    and flagged frames are COPIED (never moved/altered) into flagged_frames/.
    Fails fast if a model is unavailable. If a `cancel` Event is set mid-run, the
    sample/read loops raise Cancelled (carrying a PartialRun) at the next frame
    boundary, which the caller may finalize via finalize_partial().
    """
    video_path = Path(video_path)
    if engine is None:
        engine = LMStudioEngine(
            cfg.base_url, cfg.backend_model_id, cfg.system_prompt,
            cfg.temperature, cfg.top_k,
        )
    # verify_model_available is LM Studio-specific (not part of the Engine
    # protocol), so guard with hasattr to keep other engines pluggable.
    if hasattr(engine, "verify_model_available"):
        engine.verify_model_available()

    base, frames_dir, flagged_dir = intake(video_path, cfg)
    settings_path = base / "settings.txt"
    _write_settings_header(settings_path, video_path, cfg)

    def _log(line: str) -> None:
        """Append a timing/summary line to settings.txt and the live console."""
        with settings_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if log is not None:
            log(line)

    run_t0 = time.perf_counter()
    frames: list[SampledFrame] = []
    results: list[ReadResult] = []
    judge_list: list[ReadResult] = []
    judge_results: list[ReadResult] | None = None
    stage = "sampling"
    try:
        # --- sampling ---
        t0 = time.perf_counter()
        sample_frames(video_path, cfg, frames_dir,
                      progress=progress, cancel=cancel, sink=frames)
        if not frames:
            raise RuntimeError(f"No frames sampled from video: {video_path}")
        _log(f"sampling: {_fmt_hms(time.perf_counter() - t0)}  ({len(frames)} frames)")

        # --- reading (back-end pass) ---
        stage = "reading"
        t0 = time.perf_counter()
        read_frames(frames, engine, progress=progress, stage="reading",
                    cancel=cancel, sink=results)
        _log(f"reading: {_fmt_hms(time.perf_counter() - t0)}")

        # --- judging (separate model, after the back-end pass) ---
        if cfg.validation.enabled and cfg.validation.judge:
            stage = "judging"
            if judge_engine is None:
                judge_engine = LMStudioEngine(
                    cfg.base_url, cfg.judge_model_id, cfg.system_prompt,
                    cfg.temperature, cfg.top_k,
                )
            if cfg.validation.judge_swap_mode == "manual" and on_manual_swap is not None:
                on_manual_swap()  # block until the user loads the judge model
            if hasattr(judge_engine, "verify_model_available"):
                judge_engine.verify_model_available()
            t0 = time.perf_counter()
            read_frames(frames, judge_engine, progress=progress, stage="judging",
                        cancel=cancel, sink=judge_list)
            judge_results = judge_list
            _log(f"judging: {_fmt_hms(time.perf_counter() - t0)}")
    except Cancelled:
        partial = PartialRun(
            cfg=cfg, base=base, settings_path=settings_path,
            frames=list(frames), results=list(results),
            judge_results=(list(judge_list) if judge_list else None), stage=stage,
        )
        _log(f"CANCELLED during {stage} ({len(results)} of {len(frames)} frames read)")
        raise Cancelled(partial) from None

    # --- validation ---
    if progress is not None:
        progress("validating", 0, 1)  # switch the status label off the read stage
    t0 = time.perf_counter()
    validations = validate(results, judge_results, cfg)
    _log(f"validation: {_fmt_hms(time.perf_counter() - t0)}")

    # Copy (never move) flagged frames for human review, counting as we go.
    flagged_n = 0
    for sf, fv in zip(frames, validations):
        if fv.flagged:
            shutil.copy2(sf.frame_path, flagged_dir / sf.frame_path.name)
            flagged_n += 1
    # Report the count structurally (the GUI tally must not depend on parsing a
    # human-readable log line); the log line below is for the file/console only.
    if on_flagged is not None:
        on_flagged(flagged_n)
    _log(f"flagged: {flagged_n}")

    out_path = base / f"{video_path.stem}.xlsx"
    write_excel(video_path.stem, frames, results, validations, out_path)
    _log(f"total: {_fmt_hms(time.perf_counter() - run_t0)}")
    return out_path
