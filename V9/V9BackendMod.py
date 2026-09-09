"""V9BackendMod - swappable OCR engine(s) behind a common interface.

Reads one image, returns a normalized ReadResult. The interfacing and
validation modules never depend on which engine produced a result
(HARD RULE #7). The V9 engine talks to LM Studio's local
OpenAI-compatible API over plain HTTP (requests).
"""
from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import requests


@dataclass
class ReadResult:
    """Normalized result of reading one image. Engine-agnostic."""
    raw_text: str             # exactly what the model output
    value: str | None         # parsed/normalized reading (filled downstream)
    confidence: float | None  # None unless a trustworthy source provides it
    engine_name: str


@runtime_checkable
class Engine(Protocol):
    """Every engine reads one image and returns a ReadResult."""
    engine_name: str

    def read(self, image: str | Path | bytes) -> ReadResult: ...


def _encode_image(image: str | Path | bytes) -> str:
    """Return a data: URL for the image, from a file path or raw bytes."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = path.read_bytes()
    elif isinstance(image, bytes):
        mime = "image/png"
        data = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


class LMStudioEngine:
    """OCR engine backed by LM Studio's OpenAI-compatible chat API.

    Stateless per call (HARD RULE #6): a fresh [system, image] message list
    each read, no history. Greedy decoding (HARD RULE #8). Confidence is
    never self-reported (HARD RULE #3) - it stays None. Parsing/normalization
    is owned downstream, so `value` is left None at this layer.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        system_prompt: str,
        temperature: float = 0.0,
        top_k: int = 1,
        max_tokens: int = 32,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.engine_name = f"lmstudio:{model_id}"

    def verify_model_available(self) -> None:
        """Fail fast if the configured model id is not offered by the server.

        LM Studio silently substitutes the loaded model for an unknown id
        (returning HTTP 200), so an explicit availability check is required to
        honour HARD RULES #4 (fail fast) and #5 (explicit over inferred) - and
        to stop a mistyped judge id silently falling back to the primary model
        (which would give the judge correlated failure modes).
        """
        url = f"{self.base_url}/models"
        try:
            resp = requests.get(url, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Cannot reach LM Studio at {url}. Is the local server running? "
                f"Original error: {e}"
            ) from e
        if resp.status_code != 200:
            raise RuntimeError(
                f"LM Studio /models returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        ids = [d.get("id") for d in resp.json().get("data", [])]
        if self.model_id not in ids:
            raise RuntimeError(
                f"Model '{self.model_id}' is not available in LM Studio. "
                f"Available: {ids}. Check the model id and that it is downloaded."
            )

    def read(self, image: str | Path | bytes) -> ReadResult:
        data_url = _encode_image(image)
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": self.temperature,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        url = f"{self.base_url}/chat/completions"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            # Transport failure -> fail fast and loud (HARD RULE #4).
            raise RuntimeError(
                f"Cannot reach LM Studio at {url}. Is the local server running "
                f"with a model loaded? Original error: {e}"
            ) from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"LM Studio returned HTTP {resp.status_code} for model "
                f"'{self.model_id}': {resp.text[:300]}. Is the model id correct "
                f"and loadable?"
            )

        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"Unexpected LM Studio response shape: {resp.text[:300]}"
            ) from e

        # Backstop against silent model substitution: LM Studio echoes the model
        # that actually served the request. If it differs from what we asked for,
        # fail fast rather than trust a wrong (possibly correlated) model.
        served = data.get("model")
        if served and served.casefold() != self.model_id.casefold():
            raise RuntimeError(
                f"LM Studio served '{served}' but '{self.model_id}' was requested "
                f"(silent model substitution - the requested model is not loaded "
                f"or not available)."
            )

        # A 200 with empty/garbage text is a VALID result (stored raw, flagged
        # downstream by the format check), not a crash.
        raw_text = (content or "").strip()
        return ReadResult(
            raw_text=raw_text,
            value=None,
            confidence=None,
            engine_name=self.engine_name,
        )
