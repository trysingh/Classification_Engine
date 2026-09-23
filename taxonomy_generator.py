"""
taxonomy_generator.py
----------------------
System-2 step. Runs ONCE per dataset/domain: samples narrations from the
input CSV, asks a local frontier model via Ollama to propose a taxonomy
respecting the size caps in AppSettings, validates + trims the result,
and caches it to disk. The System-1 classifier never calls an LLM at
per-row classification time -- it only ever reads the cached taxonomy.

Requires a running Ollama instance with the configured model pulled:
    ollama pull llama3.1
"""

import json
import random
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel

from config import AppSettings


class TaxonomyResponse(BaseModel):
    """Shape enforced on the model via Ollama's structured `format` param.
    Using a real JSON schema here (not just format='json') constrains keys
    and types, not just JSON syntax -- Ollama will retry/repair internally
    to match it rather than us catching malformed output after the fact."""
    taxonomy: Dict[str, List[str]]


def _call_ollama_structured(prompt: str, schema: type[BaseModel], settings: AppSettings,
                             system: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    options: Dict[str, Any] = {"temperature": 0.0, "top_p": 1.0}
    if seed is not None:
        # Ollama honors seed on most backends; where it doesn't, the vote
        # still works -- it then measures real sampling variance rather
        # than seeded variance, which is if anything the more honest signal.
        options["seed"] = seed

    payload = {
        "model": settings.system2_model,
        "messages": [
            *([{"role": "system", "content": system}] if system else []),
            {"role": "user", "content": prompt},
        ],
        "format": schema.model_json_schema(),
        "stream": False,
        "options": options,
    }
    resp = requests.post(f"{settings.ollama_base_url}/api/chat", json=payload,
                          timeout=settings.system2_timeout_sec)
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]
    return schema.model_validate_json(raw).model_dump()


def _validate_and_trim(taxonomy: dict, settings: AppSettings) -> dict:
    """Enforce the size caps regardless of what the model actually returned."""
    others = settings.others_label
    trimmed = dict(list(taxonomy.items())[: settings.max_main_categories])

    if others not in trimmed:
        if len(trimmed) >= settings.max_main_categories:
            trimmed.pop(next(reversed(trimmed)))   # drop lowest-priority main to make room
        trimmed[others] = []

    for main, subs in trimmed.items():
        if main == others:
            trimmed[main] = []
        else:
            deduped = list(dict.fromkeys(subs))    # preserve order, drop dupes
            trimmed[main] = deduped[: settings.max_sub_categories]

    return trimmed


def generate_taxonomy(narrations: list[str], settings: AppSettings) -> dict:
    sample = random.sample(narrations, min(settings.taxonomy_sample_size, len(narrations)))
    prompt = settings.taxonomy_generation_prompt.format(
        n=len(sample),
        sample="\n".join(f"- {s}" for s in sample),
        max_main=settings.max_main_categories,
        max_sub=settings.max_sub_categories,
        others_label=settings.others_label,
    )

    result = _call_ollama_structured(prompt, TaxonomyResponse, settings, seed=settings.system2_seed)
    taxonomy = _validate_and_trim(result["taxonomy"], settings)

    settings.taxonomy_path.write_text(json.dumps(taxonomy, indent=2))
    return taxonomy


def load_or_generate_taxonomy(narrations: list[str], settings: AppSettings, force: bool = False) -> dict:
    if not force and settings.taxonomy_path.exists():
        return json.loads(settings.taxonomy_path.read_text())
    return generate_taxonomy(narrations, settings)
