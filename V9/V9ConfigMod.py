"""V9ConfigMod - explicit settings and defaults for the V9 OCR pipeline.

Pure configuration: typed defaults only. No I/O, no processing logic.
The front-end overrides these values; every other module reads them.
HARD RULE #5: configuration is explicit and surfaced - nothing inferred.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Shared system prompt - fed to BOTH the back-end and the judge (independence
# comes from using a different model, not different instructions).
SYSTEM_PROMPT = (
    "Read the number shown on the digital display in the image. "
    "Output only that number, exactly as shown, including the decimal point. "
    "No words, units, symbols, or explanation."
)

SamplingMethod = Literal["laplacian", "first_frame"]
FilterName = Literal["unfiltered", "greyscale", "contrast", "black_and_white", "adaptive"]
SwapMode = Literal["auto", "manual"]
Interpolation = Literal["nearest", "linear", "cubic"]
AdaptiveMethod = Literal["mean", "gaussian"]


@dataclass
class FilterParams:
    """Per-filter knobs. Defaults are sane starting points, tuned on real frames.

    Upscaling is adaptive + capped: small crops are enlarged toward a target to
    aid the reader, large frames are left alone or clamped, so a filter never
    explodes a frame past the model's context (which yields empty reads). Sizes
    are long-side pixels.
    """
    upscale_target_long_side: int = 1280   # enlarge small crops toward this
    upscale_max_long_side: int = 1920      # hard ceiling (>this -> empty reads observed)
    upscale_max_factor: float = 4.0        # cap interpolation on tiny crops
    interpolation: Interpolation = "cubic"
    # binary filter: None -> Otsu auto-threshold; else a fixed 0-255 cutoff
    binary_threshold: int | None = None
    # adaptive filter
    adaptive_method: AdaptiveMethod = "gaussian"
    adaptive_block_size: int = 31  # must be odd
    adaptive_c: int = 5
    # contrast filter (CLAHE)
    contrast_clip_limit: float = 2.0
    contrast_tile_grid: int = 8


@dataclass
class ValidationConfig:
    """Master toggle + independent per-check toggles (multi-select; no 'both')."""
    enabled: bool = True            # master switch for the whole post-pass
    format_check: bool = True
    range_check: bool = False       # OFF until real physical bounds are supplied
    temporal_check: bool = True
    judge: bool = True              # AI judge ON by default (doubles the read passes)
    # range check (hard bounds, in reading units)
    range_min: float | None = None
    range_max: float | None = None
    # temporal check (modified z-score over a rolling window; FLAG only)
    temporal_window: int = 5        # neighbours considered, including self
    temporal_cutoff: float = 3.5
    # judge serving
    judge_swap_mode: SwapMode = "auto"


@dataclass
class Config:
    """Top-level explicit settings for a single pipeline run."""
    # --- sampling ---
    samples_per_second: float = 1.0
    sampling_method: SamplingMethod = "laplacian"

    # --- filtering ---
    filter_name: FilterName = "greyscale"
    filter_params: FilterParams = field(default_factory=FilterParams)

    # --- reading / format ---
    format_mask: str = "NN.N"
    allow_negative: bool = False
    system_prompt: str = SYSTEM_PROMPT

    # --- LM Studio connection ---
    base_url: str = "http://127.0.0.1:1234/v1"
    backend_model_id: str = "qwen/qwen3-vl-4b"
    judge_model_id: str = "minicpm-v-4_5"

    # --- deterministic decoding (HARD RULE #8) ---
    temperature: float = 0.0
    top_k: int = 1

    # --- validation ---
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # --- paths ---
    # Single output root. Relative names are resolved against the V9 project
    # root by the interfacing module (see _outputs_root), so the location does
    # not depend on the launch directory. Per-video results live in
    # outputs_dir/<video_name>/.
    outputs_dir: str = "Outputs"
