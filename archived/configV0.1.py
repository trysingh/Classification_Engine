"""
config.py
---------
Single source of truth for the hybrid classifier: model choices, Ollama
connection, taxonomy size limits, and every prompt template.

Edit values here, or override via a `.env` file / environment variables
prefixed `OPENJEV_` (e.g. OPENJEV_SYSTEM2_MODEL=mistral). Nothing else
in the codebase should hardcode a prompt or a category limit.
"""

from typing import Literal
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="OPENJEV_", extra="ignore")

    # --- Backend switch: which System-1 engine actually runs classify_hierarchical ---
    # "causal_lm" = current Qwen2.5 logit read-off (needs permutation debiasing).
    # "laya"      = Convai's Laya, non-autoregressive option-marker scoring
    #               (structurally immune to the position bias, ~33ms/call).
    # Flip back any time via OPENJEV_CLASSIFIER_BACKEND=causal_lm or --backend causal_lm.
    classifier_backend: Literal["causal_lm", "laya"] = "causal_lm"

    # --- Laya backend ---
    laya_model_name: str = "convaiinnovations/laya"

    # --- System 1: fast path, local logit read-off, runs on every row ---
    system1_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # --- System 2: slow path, taxonomy design (+ optional future use), via Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    system2_model: str = "gemma4:31b-cloud"
    system2_timeout_sec: int = 120
    system2_seed: int | None = 42     # None = let Ollama sample freely each regeneration

    # --- Taxonomy shape constraints, enforced on whatever System 2 returns ---
    max_main_categories: int = 7
    max_sub_categories: int = 10
    taxonomy_sample_size: int = 40          # narrations sampled to design the taxonomy
    taxonomy_path: Path = Path("taxonomy.json")
    others_label: str = "Others"

    # Small models show strong positional bias in MCQ-style logit read-off
    # (systematically favoring e.g. the first/last option regardless of
    # content). Running each Choice this many times with shuffled option
    # order and averaging probability per label cancels that bias out.
    debias_permutations: int = 3

    # --- Prompt templates ---
    taxonomy_generation_prompt: str = (
        "You are designing a cost-classification taxonomy for ERP narrations describing "
        "software-system costs in a banking enterprise.\n\n"
        "Here is a sample of {n} real narrations from the ledger:\n{sample}\n\n"
        "Design a taxonomy with AT MOST {max_main} main categories and AT MOST {max_sub} "
        "sub-categories inside each main category. Main categories must be broad enough to "
        "cover the whole sample (e.g. infrastructure, network, licensing, operations, "
        "governance/compliance, consulting/professional services, maintenance/support, and "
        "similar groupings you infer from the sample). Always include exactly one main "
        "category literally named \"{others_label}\" as a catch-all, with an EMPTY "
        "sub-category list (its sub-categories are produced dynamically per narration later, "
        "not fixed up front).\n\n"
        "Respond with ONLY a JSON object, no prose, no markdown fences, in this exact shape:\n"
        '{{"taxonomy": {{"Main Category A": ["Sub 1", "Sub 2"], "Main Category B": ["Sub 1"], '
        '"{others_label}": []}}}}'
    )

    main_category_question: str = (
        "What is the main cost category of this ERP narration for a banking software system? "
        "Choose '{others_label}' only if none of the other categories fit."
    )
    sub_category_question: str = (
        "Within '{main_category}', what is the specific sub-category of this ERP narration?"
    )
    dynamic_label_prompt: str = (
        "ERP cost narration: {narration}\n"
        "In 2-4 words, name the most specific cost sub-category this belongs to:\n"
        "Sub-category:"
    )


settings = AppSettings()