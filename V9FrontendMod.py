"""V9FrontendMod - thin Gradio GUI over the interfacing module.

Collects a video and settings, hands off to process_video, shows progress, and
offers the Excel download. Contains NO processing logic of its own (HARD RULE
#7) - it only builds a Config from the widgets and calls the orchestrator.

Progress is shown as ONE static text bar in the Status box, refreshed only when
a frame actually completes (event-driven via a queue). There is deliberately no
animated `gr.Progress` widget and `show_progress="hidden"` on every event - an
earlier version's animated bar painted an overlay on every output box, flickered,
and hammered the iGPU (which slowed the dGPU model reads).
"""

# Launch the GUI. Open the printed URL in a browser:
#   http://127.0.0.1:7860
# On launch: if a server is already on that port it is reused (no second launch);
# otherwise the server starts and your default browser opens. (See __main__.)

from __future__ import annotations

import queue
import socket
import threading
import time

import gradio as gr

from V9ConfigMod import Config
from V9InterfacingMod import process_video, finalize_partial, check_gpu, Cancelled, _fmt_hms

# Module-level coordination (single local user, one run at a time).
_SWAP = threading.Event()        # manual judge-swap gate
_CANCEL = threading.Event()      # cancel flag for the active run
_PARTIAL: dict[str, object] = {"val": None}  # last cancelled run's PartialRun
_SENTINEL = object()             # "worker finished" marker on the queue

FILTERS = ["unfiltered", "greyscale", "contrast", "black_and_white", "adaptive"]

# Friendly stage labels for the status bar.
_STAGE_LABEL = {
    "starting": "Starting…",
    "sampling": "Sampling…",
    "reading": "Reading…",
    "judging": "Judging…",
    "validating": "Validating…",
}

_HOST, _PORT = "127.0.0.1", 7860


def _bar(stage_label: str, cur: int, total: int, elapsed: float) -> str:
    """One static (non-animated) text progress bar for the Status box."""
    total = max(0, int(total))
    cur = max(0, int(cur))
    pct = min(1.0, cur / total) if total else 0.0
    filled = int(round(pct * 20))
    track = "█" * filled + "░" * (20 - filled)
    return f"{stage_label}  [{track}]  {int(pct * 100)}%   {cur}/{total}   |   {_fmt_hms(elapsed)}"


def _build_config(
    samples_per_second, sampling_method, filter_name, format_mask, allow_negative,
    system_prompt, val_enabled, format_check, range_check, temporal_check, judge,
    range_min, range_max, temporal_window, temporal_cutoff, judge_swap_mode,
    base_url, backend_model_id, judge_model_id,
) -> Config:
    """Map the widget values onto an explicit Config (HARD RULE #5)."""
    cfg = Config()
    cfg.samples_per_second = float(samples_per_second)
    cfg.sampling_method = sampling_method
    cfg.filter_name = filter_name
    cfg.format_mask = format_mask
    cfg.allow_negative = bool(allow_negative)
    cfg.system_prompt = system_prompt
    cfg.base_url = base_url
    cfg.backend_model_id = backend_model_id
    cfg.judge_model_id = judge_model_id
    v = cfg.validation
    v.enabled = bool(val_enabled)
    v.format_check = bool(format_check)
    v.range_check = bool(range_check)
    v.temporal_check = bool(temporal_check)
    v.judge = bool(judge)
    v.range_min = float(range_min) if range_min not in (None, "") else None
    v.range_max = float(range_max) if range_max not in (None, "") else None
    v.temporal_window = int(temporal_window)
    v.temporal_cutoff = float(temporal_cutoff)
    v.judge_swap_mode = judge_swap_mode
    return cfg


def _run(video, *settings):
    """Run the pipeline in a worker thread, streaming status to the UI.

    Yields (status, console, flagged, excel_file, continue_btn, cancel_btn,
    discard_btn, save_btn, run_btn). The worker runs process_video; progress
    events arrive on a queue and the generator yields once per event (no timer,
    no animated widget). On cancellation the Discard / Save-partial buttons appear.
    """
    hide = gr.update(visible=False)
    show = gr.update(visible=True)
    cancel_live = gr.update(visible=True, interactive=True, value="Cancel Job")
    run_off = gr.update(interactive=False)
    run_on = gr.update(interactive=True)

    if not video:
        yield ("No video selected.", "", "**Flagged:** 0", None,
               hide, hide, hide, hide, run_on)
        return

    cfg = _build_config(*settings)
    state = {"flagged": 0, "log": [], "out": None, "error": None,
             "cancelled": False}
    q: "queue.Queue" = queue.Queue()
    _SWAP.clear()
    _CANCEL.clear()
    _PARTIAL["val"] = None

    def prog(stage, cur, total):
        q.put((stage, cur, total))

    def log_cb(line):
        state["log"].append(line)
        q.put(("__log__", 0, 0))  # wake the generator so the console refreshes

    def flag_cb(n):
        state["flagged"] = n

    def manual_gate():
        q.put(("__await_swap__", 0, 0))
        _SWAP.wait()
        _SWAP.clear()

    def worker():
        try:
            state["out"] = str(process_video(
                video, cfg, on_manual_swap=manual_gate, progress=prog,
                on_flagged=flag_cb, log=log_cb, cancel=_CANCEL))
        except Cancelled as c:
            _PARTIAL["val"] = c.partial
            state["cancelled"] = True
        except Exception as exc:  # surface failures loudly (HARD RULE #4)
            state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            q.put(_SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    start = time.time()
    cur = total = 0
    stage_label = "Starting…"
    await_swap = False
    done = False
    while not done:
        first = q.get()                     # block until something happens
        items = [first]
        try:                                # drain queued events, coalesce to latest
            while True:
                items.append(q.get_nowait())
        except queue.Empty:
            pass
        if any(it is _SENTINEL for it in items):
            done = True
        if any(it == ("__await_swap__", 0, 0) for it in items):
            await_swap = True
        progs = [it for it in items
                 if it is not _SENTINEL and it[0] not in ("__await_swap__", "__log__")]
        if progs:
            st, cur, total = progs[-1]
            stage_label = _STAGE_LABEL.get(st, st)
            await_swap = False              # real progress resumed
        if done:
            break
        elapsed = time.time() - start
        console = "\n".join(state["log"])
        flag_md = f"**Flagged:** {state['flagged']}"
        if await_swap:
            yield ("Back-end pass complete. Load the judge model (MiniCPM) in "
                   "LM Studio, then click Continue.", console, flag_md, None,
                   show, cancel_live, hide, hide, run_off)
        else:
            yield (_bar(stage_label, cur, total, elapsed), console, flag_md, None,
                   hide, cancel_live, hide, hide, run_off)

    # --- terminal states ---
    elapsed = time.time() - start
    console = "\n".join(state["log"])
    flag_md = f"**Flagged:** {state['flagged']}"
    if state["cancelled"]:
        yield ("Cancelled. Keep the partial results, or discard?",
               console, flag_md, None, hide, hide, show, show, run_on)
    elif state["error"]:
        yield (f"ERROR - {state['error']}", console, flag_md, None,
               hide, hide, hide, hide, run_on)
    else:
        yield (f"Done in {_fmt_hms(elapsed)}.", console, flag_md, state["out"],
               hide, hide, hide, hide, run_on)


def _continue():
    """Signal the blocked judge-swap gate to proceed."""
    _SWAP.set()
    return gr.update(visible=False)


def _cancel():
    """Request cancellation; also release a blocked manual-swap gate."""
    _CANCEL.set()
    _SWAP.set()
    return gr.update(interactive=False, value="Cancelling…")


def _discard():
    """Drop the cancelled run's partial results - no spreadsheet."""
    _PARTIAL["val"] = None
    return ("Cancelled - discarded. No spreadsheet written.", None,
            gr.update(visible=False), gr.update(visible=False))


def _save_partial():
    """Write a clearly-stamped PARTIAL Excel from the cancelled run."""
    p = _PARTIAL["val"]
    if p is None:
        return ("Nothing to save.", None,
                gr.update(visible=False), gr.update(visible=False))
    try:
        out = str(finalize_partial(p))
        msg = f"Saved PARTIAL results -> {out}"
    except Exception as e:  # fail loud
        out, msg = None, f"ERROR saving partial: {type(e).__name__}: {e}"
    _PARTIAL["val"] = None
    return (msg, out, gr.update(visible=False), gr.update(visible=False))


def _check_gpu(*settings):
    """Probe whether LM Studio is using the dGPU (green) or iGPU/CPU (yellow)."""
    try:
        cfg = _build_config(*settings)
        return check_gpu(cfg)
    except Exception as e:
        return f"⚠️ GPU check failed: {type(e).__name__}: {e}"


def build_ui() -> gr.Blocks:
    cfg0 = Config()
    with gr.Blocks(title="Video OCR V9") as demo:  # browser tab title
        gr.Markdown("# Video OCR Software V9")
        with gr.Row():
            with gr.Column():
                video = gr.File(label="Input video",
                                file_types=[".mp4", ".mov", ".avi", ".mkv"])
                with gr.Accordion("Sampling", open=False):
                    sps = gr.Number(
                        value=1.0, label="Samples per second",
                        info="How many frames to read per second of video. 1.0 = one reading each second.")
                    smethod = gr.Radio(
                        ["laplacian", "first_frame"], value="laplacian",
                        label="Sampling method",
                        info="laplacian: pick the sharpest frame in each second (best for motion blur). "
                             "first_frame: just take the first frame of each second (faster).")
                with gr.Accordion("Filtering", open=False):
                    fname = gr.Dropdown(
                        FILTERS, value="greyscale", label="Filter",
                        info="Image processing applied before reading; the saved frame IS what the model sees. "
                             "greyscale (default), contrast (CLAHE), black_and_white (threshold), "
                             "adaptive (local threshold), or unfiltered (original).")
                with gr.Accordion("Reading / format", open=False):
                    mask = gr.Textbox(
                        value="NN.N", label="Reading format mask",
                        info="Expected number shape: N = a digit. 'NN.N' = two digits, a point, one decimal.")
                    neg = gr.Checkbox(
                        value=False, label="Allow negative",
                        info="Accept a leading minus sign in readings.")
                    sysp = gr.Textbox(
                        value=cfg0.system_prompt, lines=3,
                        label="System prompt (sent to both models)",
                        info="Instruction given to the vision model(s) for every frame.")
                with gr.Accordion("Validation", open=True):
                    ven = gr.Checkbox(
                        value=True, label="Validation enabled (master)",
                        info="Master switch for all checks below. Off = read only, no flagging.")
                    fcheck = gr.Checkbox(
                        value=True, label="Format check",
                        info="Flag readings that do not match the format mask.")
                    rcheck = gr.Checkbox(
                        value=False, label="Range check",
                        info="Flag readings outside the min/max below. Leave off until real bounds are set.")
                    with gr.Row():
                        rmin = gr.Number(value=None, label="Range min",
                                         info="Hard lower bound (reading units).")
                        rmax = gr.Number(value=None, label="Range max",
                                         info="Hard upper bound (reading units).")
                    tcheck = gr.Checkbox(
                        value=True, label="Temporal check",
                        info="Flag readings that jump sharply from their time-neighbours. Flags only, never edits.")
                    with gr.Row():
                        twin = gr.Number(value=5, label="Temporal window",
                                         info="How many neighbouring seconds the check compares against.")
                        tcut = gr.Number(value=3.5, label="Temporal cutoff",
                                         info="Sensitivity: lower = flags more, higher = flags fewer.")
                    jcheck = gr.Checkbox(
                        value=True, label="AI judge (MiniCPM)",
                        info="A second, different model re-reads every frame; disagreements are flagged. "
                             "Doubles reading time. On by default.")
                    jswap = gr.Radio(
                        ["auto", "manual"], value="auto", label="Judge swap mode",
                        info="auto: LM Studio loads the judge model on demand. "
                             "manual: you load it yourself and click Continue.")
                with gr.Accordion("LM Studio connection", open=False):
                    burl = gr.Textbox(value=cfg0.base_url, label="Base URL",
                                      info="LM Studio local server address.")
                    bmid = gr.Textbox(value=cfg0.backend_model_id,
                                      label="Back-end model id",
                                      info="Model id of the primary reader.")
                    jmid = gr.Textbox(value=cfg0.judge_model_id, label="Judge model id",
                                      info="Model id of the independent judge (must differ from the reader).")
                run_btn = gr.Button("Run", variant="primary")
                check_btn = gr.Button("Check GPU (dGPU vs iGPU)")
                gpu_status = gr.Markdown("", label="GPU check")
            with gr.Column():
                status = gr.Textbox(label="Status", interactive=False)
                flagged = gr.Markdown("**Flagged:** 0")
                console = gr.Textbox(label="Run log", lines=8, interactive=False,
                                     info="Per-stage timing; exported into settings.txt at the end of the run.")
                cont_btn = gr.Button("Continue (judge model loaded)", visible=False)
                cancel_btn = gr.Button("Cancel Job", variant="stop", visible=False)
                with gr.Row():
                    discard_btn = gr.Button("Discard partial", visible=False)
                    save_btn = gr.Button("Save partial results", visible=False)
                out_file = gr.File(label="Excel output")

        settings = [sps, smethod, fname, mask, neg, sysp, ven, fcheck, rcheck,
                    tcheck, jcheck, rmin, rmax, twin, tcut, jswap, burl, bmid, jmid]
        run_outputs = [status, console, flagged, out_file,
                       cont_btn, cancel_btn, discard_btn, save_btn, run_btn]
        # show_progress="hidden" everywhere: no per-component overlay bars (the
        # source of the flicker / iGPU hog). Our own text bar lives in Status.
        run_btn.click(_run, [video] + settings, run_outputs, show_progress="hidden")
        cont_btn.click(_continue, None, cont_btn, show_progress="hidden")
        cancel_btn.click(_cancel, None, cancel_btn, show_progress="hidden")
        discard_btn.click(_discard, None, [status, out_file, discard_btn, save_btn],
                          show_progress="hidden")
        save_btn.click(_save_partial, None, [status, out_file, discard_btn, save_btn],
                       show_progress="hidden")
        check_btn.click(_check_gpu, settings, gpu_status, show_progress="hidden")
    return demo


def _port_in_use(host: str, port: int) -> bool:
    """True if something is already listening on host:port (a server is up)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


if __name__ == "__main__":
    url = f"http://{_HOST}:{_PORT}"
    # If a server is already serving this port (e.g. a previous run), reuse it:
    # don't launch a second one (which would crash with "port in use") and don't
    # pop a browser. Otherwise start the server and open the default browser.
    if _port_in_use(_HOST, _PORT):
        print(f"Video OCR V9 already running at {url} - reusing it (not launching a second server).")
    else:
        print(f"Starting Video OCR V9 at {url}")
        build_ui().launch(server_name=_HOST, server_port=_PORT, inbrowser=True)
