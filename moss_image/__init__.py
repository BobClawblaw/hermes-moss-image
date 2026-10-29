"""MOSS-Image on a TensorFold server (``POST /v1/images/generations``).

The image tool passes only a prompt and an aspect ratio, so the options this model takes are read from the
prompt and removed from it: ``WIDTHxHEIGHT`` (``512x512``, ``1344 x 768``), ``N steps``, ``seed N`` and
``guidance X``. Without a size the aspect ratio picks 1024x1024, 1536x1024 or 1024x1536.
Endpoint: ``image_gen.moss.base_url`` (default ``http://127.0.0.1:8899/v1``); model: ``image_gen.moss.model``
(default ``moss-image-base``); an optional key from the env var named by ``image_gen.moss.key_env``."""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Optional

from agent.image_gen_provider import DEFAULT_ASPECT_RATIO, resolve_aspect_ratio, success_response
from agent.secret_scope import get_secret
from plugins.image_gen._common import (
    StaticImageGenProvider, error_factory, load_image_gen_config, materialize_image, prompt_required_error)

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8899/v1"
DEFAULT_MODEL = "moss-image-base"
ASPECT_SIZES = {"square": (1024, 1024), "landscape": (1536, 1024), "portrait": (1024, 1536)}
# The server's limits: multiples of 16, 1536 a side, 1536 x 1024's pixels, 1 to 200 steps.
MAX_SIDE, MAX_PIXELS, MAX_STEPS = 1536, 1536 * 1024, 200

_SIZE = re.compile(r"\b(\d{2,4})\s*[x×]\s*(\d{2,4})\b(?:\s*(?:px|pixels?))?", re.I)
_STEPS = re.compile(r"\b(\d{1,3})\s*(?:steps?|iterations?)\b", re.I)
_SEED = re.compile(r"\bseed\s*[:=]?\s*(\d{1,19})\b", re.I)
_GUIDANCE = re.compile(r"\b(?:guidance|cfg)\s*(?:scale)?\s*[:=]?\s*(\d+(?:\.\d+)?)\b", re.I)


def parse_options(prompt: str) -> tuple[str, dict[str, Any]]:
    """The prompt with its options removed, and the options found (size, steps, seed, guidance)."""
    found: dict[str, Any] = {}
    for key, pattern, value in (
        ("size", _SIZE, lambda m: (int(m.group(1)), int(m.group(2)))),
        ("steps", _STEPS, lambda m: int(m.group(1))),
        ("seed", _SEED, lambda m: int(m.group(1))),
        ("guidance", _GUIDANCE, lambda m: float(m.group(1))),
    ):
        m = pattern.search(prompt)
        if m:
            found[key] = value(m)
            prompt = prompt[:m.start()] + prompt[m.end():]
    prompt = re.sub(r"\s{2,}", " ", re.sub(r"\s+([,.;:])", r"\1", prompt)).strip(" ,;:-")
    return prompt, found


def fit_size(width: int, height: int) -> tuple[Optional[tuple[int, int]], str]:
    """A size the server takes (each side rounded to a multiple of 16), or None and why not."""
    w, h = max(16, round(width / 16) * 16), max(16, round(height / 16) * 16)
    if w > MAX_SIDE or h > MAX_SIDE or w * h > MAX_PIXELS:
        return None, f"{width}x{height} is past the server's limits: at most {MAX_SIDE} a side and {MAX_PIXELS:,} pixels (1536x1024)"
    return (w, h), ""


def _endpoint() -> tuple[str, str, str]:
    cfg = load_image_gen_config("moss")
    url = str(cfg.get("base_url") or DEFAULT_URL).strip().rstrip("/")
    model = str(cfg.get("model") or DEFAULT_MODEL).strip()
    key_env = str(cfg.get("key_env") or "").strip()
    key = (get_secret(key_env) if key_env else None) or ""
    return url, model, key


class MossImageGenProvider(StaticImageGenProvider):
    """MOSS-Image served by TensorFold, with exact sizes and sampling options from the prompt."""

    provider_id = "moss"
    label = "MOSS-Image (local TensorFold)"
    models = {DEFAULT_MODEL: {"display": "MOSS-Image Base", "speed": "~48 s at 1024x1024, 50 steps",
                              "strengths": "exact sizes, steps and seed from the prompt", "price": "local"}}
    default_model_id = DEFAULT_MODEL
    price = "local"

    def is_available(self) -> bool:
        return True

    def capabilities(self) -> dict[str, Any]:
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *,
                 image_url: Optional[str] = None, reference_image_urls: Optional[list[str]] = None,
                 **kwargs: Any) -> dict[str, Any]:
        aspect = resolve_aspect_ratio(aspect_ratio)
        text, opts = parse_options((prompt or "").strip())
        if not text:
            return prompt_required_error("moss", aspect)
        url, model, key = _endpoint()
        fail = error_factory("moss", aspect, model=model, prompt=text)
        if image_url or reference_image_urls:
            return fail("MOSS-Image makes new pictures only; it cannot edit or use reference images", "unsupported")
        if "size" in opts:
            size, why = fit_size(*opts["size"])
            if size is None:
                return fail(why, "invalid_argument")
        else:
            size = ASPECT_SIZES.get(aspect, ASPECT_SIZES["square"])
        body: dict[str, Any] = {"model": model, "prompt": text, "size": f"{size[0]}x{size[1]}", "n": 1,
                                "response_format": "b64_json"}
        if "steps" in opts:
            if not 1 <= opts["steps"] <= MAX_STEPS:
                return fail(f"steps must be 1 through {MAX_STEPS}", "invalid_argument")
            body["steps"] = opts["steps"]
        for k in ("seed", "guidance"):
            if k in opts:
                body[k] = opts[k]
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(f"{url}/images/generations", data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=900) as r:
                reply = json.load(r)
        except urllib.error.HTTPError as exc:
            try:
                message = json.loads(exc.read() or b"{}").get("error", {}).get("message") or str(exc)
            except Exception:
                message = str(exc)
            kind = "busy" if exc.code == 503 else "auth_required" if exc.code == 401 else "api_error"
            return fail(f"MOSS-Image server: {message}", kind)
        except Exception as exc:
            logger.debug("MOSS-Image request failed", exc_info=True)
            return fail(f"MOSS-Image server at {url} could not be reached: {exc}", "api_error")
        data = reply.get("data") or []
        if not data or not data[0].get("b64_json"):
            return fail("MOSS-Image returned no picture", "empty_response")
        image_ref, err = materialize_image(data[0]["b64_json"], None, prefix=f"moss_{size[0]}x{size[1]}",
                                           label="MOSS-Image", provider="moss", model=model, prompt=text,
                                           aspect=aspect, log=logger)
        if err:
            return err
        extra = {"size": body["size"], **{k: body[k] for k in ("steps", "seed", "guidance") if k in body}}
        return success_response(image=image_ref, model=model, prompt=text, aspect_ratio=aspect, provider="moss",
                                modality="text", extra=extra)


def register(ctx) -> None:
    """Plugin entry point: add the MOSS-Image provider to the image-generation registry."""
    ctx.register_image_gen_provider(MossImageGenProvider())
