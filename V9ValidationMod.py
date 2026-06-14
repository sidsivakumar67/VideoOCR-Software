"""V9ValidationMod - reading normalization and the toggleable validation post-pass.

`normalize()` is the always-on parser: it turns raw model output into a value
matching the configured mask and runs for every reading, so the Excel
"parsed value" column is always populated. `validate()` then runs the toggleable
post-pass over all readings - format, range, and temporal checks plus the
independent AI judge - returning one FrameValidation per reading.

HARD RULE #2: nothing here corrects or fabricates a reading. Checks only FLAG;
the parser only normalizes structure (returns None when it cannot, and reports
an inserted decimal) so the caller can flag it. Confidence comes solely from the
judge's agreement (HARD RULE #3).
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from V9ConfigMod import Config
from V9BackendMod import ReadResult


def _parse_mask(mask: str) -> tuple[int, int]:
    """Parse a mask like 'NN.N' into (integer_digits, decimal_digits)."""
    mask = mask.strip()
    if "." in mask:
        left, _, right = mask.partition(".")
        return left.count("N"), right.count("N")
    return mask.count("N"), 0


def normalize(
    raw_text: str, mask: str = "NN.N", allow_negative: bool = False
) -> tuple[str | None, bool]:
    """Normalize raw model output to the mask.

    Returns (value, inserted):
      value    - normalized numeric string matching the mask, or None if it
                 cannot be normalized (the caller flags None).
      inserted - True if a missing decimal point was inserted (caller flags it).

    Rules:
      - Clean: comma -> dot, then extract the first numeric token.
      - A decimal point that IS present is NEVER relocated; the value must
        already fit the mask exactly, else -> None.
      - With no point and a digit count equal to the mask total, insert the
        point at the mask boundary -> value, inserted=True.
      - Anything else -> None. Negatives only when allow_negative.
    """
    int_digits, dec_digits = _parse_mask(mask)
    s = (raw_text or "").strip().replace(",", ".")
    match = re.search(r"-?\d+\.?\d*", s)
    if not match:
        return None, False

    token = match.group(0)
    negative = token.startswith("-")
    if negative and not allow_negative:
        return None, False
    sign = "-" if negative else ""
    body = token[1:] if negative else token

    if "." in body:
        left, _, right = body.partition(".")
        # Never relocate an existing point - it must already fit the mask.
        if left and right and len(left) == int_digits and len(right) == dec_digits:
            return sign + left + "." + right, False
        return None, False

    # No point present.
    digits = body
    if dec_digits > 0 and len(digits) == int_digits + dec_digits:
        return sign + digits[:int_digits] + "." + digits[int_digits:], True
    if dec_digits == 0 and len(digits) == int_digits:
        return sign + digits, False
    return None, False


# --------------------------------------------------------------------------
# Validation post-pass (Step 4)
# --------------------------------------------------------------------------
@dataclass
class FrameValidation:
    """Per-reading validation outcome. Pure data; never alters the reading.

    Empty string for a check means that check is off. The reading model's value
    is never changed - only flagged (HARD RULE #2).
    """
    parsed_value: str | None
    format_check: str = ""     # "" off | OK | inserted | fail
    range_check: str = ""      # "" off | OK | fail | skip
    temporal_check: str = ""   # "" off | OK | outlier | skip
    judge: str = ""            # "" off | agree | disagree
    flagged: bool = False
    flag_reasons: list[str] = field(default_factory=list)
    confidence: float | None = None
    judge_raw: str | None = None   # the judge's raw read, kept for debugging disagreements


def _temporal_outliers(
    values: list[float | None], window: int, cutoff: float
) -> list[bool]:
    """Flag points that deviate sharply from their time-neighbours.

    Modified z-score that centres on the LOCAL median (ramp-safe: a steady trend
    sits at its own median) but scales by the GLOBAL median absolute deviation of
    the whole series (a stable, scale-free dispersion). This way an ordinary
    staircase step is a small z, while a true spike (e.g. 24.5 read as 99.9) is a
    huge z. FLAG only - nothing is altered or removed (HARD RULE #2).
    """
    n = len(values)
    out = [False] * n
    nums = [v for v in values if v is not None]
    if len(nums) < 3:
        return out

    gmed = statistics.median(nums)
    gdevs = [abs(v - gmed) for v in nums]
    scale = statistics.median(gdevs)
    if scale == 0:  # perfectly stable series -> fall back to mean abs deviation
        scale = sum(gdevs) / len(gdevs)

    half = max(1, window // 2)
    for i in range(n):
        xi = values[i]
        if xi is None:
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        neigh = [v for v in values[lo:hi] if v is not None]
        if len(neigh) < 2:
            continue
        lmed = statistics.median(neigh)
        if scale == 0:
            out[i] = xi != lmed  # whole series constant -> any change is anomalous
        else:
            out[i] = (0.6745 * abs(xi - lmed) / scale) > cutoff
    return out


def validate(
    backend_results: list[ReadResult],
    judge_results: list[ReadResult] | None,
    cfg: Config,
) -> list[FrameValidation]:
    """Run the toggleable post-pass over all readings (pure - no frame I/O).

    parsed_value is always populated by the always-on normalize(); checks run
    only when enabled. The judge compares an independent model's read, never the
    same model. Nothing is corrected, smoothed, or removed (HARD RULE #2).
    """
    v = cfg.validation
    mask, neg = cfg.format_mask, cfg.allow_negative

    # Always-on parse -> parsed_value for every row.
    parsed: list[str | None] = []
    inserted: list[bool] = []
    for r in backend_results:
        val, ins = normalize(r.raw_text, mask, neg)
        parsed.append(val)
        inserted.append(ins)

    out = [FrameValidation(parsed_value=parsed[i]) for i in range(len(backend_results))]
    if not v.enabled:
        return out

    temporal_out: list[bool] | None = None
    if v.temporal_check:
        nums = [float(p) if p is not None else None for p in parsed]
        temporal_out = _temporal_outliers(nums, v.temporal_window, v.temporal_cutoff)

    for i in range(len(backend_results)):
        fv = out[i]
        val = parsed[i]

        if v.format_check:
            if val is None:
                fv.format_check = "fail"
                fv.flag_reasons.append("format_fail")
            elif inserted[i]:
                fv.format_check = "inserted"
                fv.flag_reasons.append("decimal_inserted")
            else:
                fv.format_check = "OK"

        if v.range_check:
            if val is None:
                fv.range_check = "skip"
            else:
                x = float(val)
                if v.range_min is not None and x < v.range_min:
                    fv.range_check = "fail"
                    fv.flag_reasons.append("range_below")
                elif v.range_max is not None and x > v.range_max:
                    fv.range_check = "fail"
                    fv.flag_reasons.append("range_above")
                else:
                    fv.range_check = "OK"

        if v.temporal_check and temporal_out is not None:
            if val is None:
                fv.temporal_check = "skip"
            elif temporal_out[i]:
                fv.temporal_check = "outlier"
                fv.flag_reasons.append("temporal_outlier")
            else:
                fv.temporal_check = "OK"

        if v.judge and judge_results is not None:
            jr = judge_results[i]
            fv.judge_raw = jr.raw_text
            jval, _ = normalize(jr.raw_text, mask, neg)
            if val is not None and jval is not None and val == jval:
                fv.judge = "agree"
                fv.confidence = 1.0
            else:
                fv.judge = "disagree"
                fv.confidence = 0.0
                fv.flag_reasons.append("judge_disagree")

        fv.flagged = len(fv.flag_reasons) > 0

    return out
