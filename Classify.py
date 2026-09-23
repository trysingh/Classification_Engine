"""
open_jev_zero_shot.py
----------------------
Hybrid System-1 / System-2 ERP cost classifier.

System 2 (Ollama / local frontier model): runs ONCE per dataset to design
a taxonomy (main + sub categories, size-capped, with a catch-all
'Others') from a sample of narrations. See taxonomy_generator.py.

System 1 (small local causal LM): runs on EVERY row, reading candidate
label logits off a single forward pass (Jev/OpenJev pattern) for the
main category, then the sub-category. Only rows landing in 'Others' get
one extra short (<=6 token) local generation to produce a specific label
instead of the static word 'Others' -- still no call to System 2.

Prompts, model names and taxonomy limits all live in config.py.
"""

import time
import random
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import AppSettings, settings as default_settings
from taxonomy_generator import load_or_generate_taxonomy


class OpenJevClassifier:
    """System-1 fast path: zero-shot Choice via logit read-off, one forward pass per question."""

    def __init__(self, taxonomy: dict, settings: AppSettings, device: str | None = None):
        self.taxonomy = taxonomy
        self.settings = settings
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.system1_model,
            trust_remote_code=True,
        )


        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.system1_model, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()
        print(f"[OpenJevClassifier] model={settings.system1_model} device={self.device} dtype={dtype}")

    def _build_prompt(self, state: str, question: str, labels: list[str]) -> str:
        options = "\n".join(f"{chr(65 + i)}. {label}" for i, label in enumerate(labels))
        return (
            f"Context: {state}\n"
            f"Question: {question}\n"
            f"Options:\n{options}\n"
            f"Answer with only the letter of the best option.\n"
            f"Answer:"
        )

    @torch.inference_mode()
    def classify(self, state: str, question: str, labels: list[str]) -> dict:
        prompt = self._build_prompt(state, question, labels)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        letters = [chr(65 + i) for i in range(len(labels))]
        candidate_ids = [self.tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]

        start = time.perf_counter()
        logits = self.model(**inputs).logits[0, -1]        # single forward pass
        latency_ms = (time.perf_counter() - start) * 1000

        candidate_logits = logits[candidate_ids]
        probs = torch.softmax(candidate_logits, dim=-1)

        return {
            "prompt_tokens": inputs["input_ids"].shape[-1],
            "latency_ms": round(latency_ms, 2),
            "choice": labels[int(probs.argmax())],
            "probabilities": {labels[i]: round(float(probs[i]), 4) for i in range(len(labels))},
        }

    @staticmethod
    def _top3(probabilities: dict) -> str:
        ranked = sorted(probabilities.items(), key=lambda x: -x[1])[:3]
        # Extract the name of the highest-scoring label
        top_ranked = ranked[0][0]
        top3_distribution = " | ".join(f"{label} ({p:.1%})" for label, p in ranked)
                            
        return top3_distribution, top_ranked

    def classify_debiased(self, state: str, question: str, labels: list[str]) -> dict:
        """Runs classify() N times with the option order shuffled each time and
        averages probability mass per LABEL (not per position). Cancels out
        the positional bias small models show in MCQ-style logit read-off,
        where a fixed first/last option can dominate regardless of content."""
        n = self.settings.debias_permutations
        rng = random.Random(42)     # fixed seed: reproducible per row, still varies the order
        agg = {label: 0.0 for label in labels}
        total_latency_ms, total_prompt_tokens = 0.0, 0

        for _ in range(n):
            shuffled = labels[:]
            rng.shuffle(shuffled)
            result = self.classify(state, question, shuffled)
            total_latency_ms += result["latency_ms"]
            total_prompt_tokens += result["prompt_tokens"]
            for label, p in result["probabilities"].items():
                agg[label] += p

        agg = {label: round(p / n, 4) for label, p in agg.items()}
        return {
            "prompt_tokens": total_prompt_tokens,
            "latency_ms": round(total_latency_ms, 2),
            "choice": max(agg, key=agg.get),
            "probabilities": agg,
        }

    @torch.inference_mode()
    def generate_dynamic_label(self, narration: str, max_new_tokens: int = 6) -> str:
        """Only path in this classifier that generates text; used exclusively for
        the 'Others' branch, where a fixed-choice logit read-off can't help."""
        prompt = self.settings.dynamic_label_prompt.format(narration=narration)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        return generated.strip().split("\n")[0].strip(" .") or "Unclassified"

    def classify_hierarchical(self, narration: str) -> dict:
        start = time.perf_counter()
        others = self.settings.others_label

        main_labels = list(self.taxonomy.keys())
        main_q = self.settings.main_category_question.format(others_label=others)
        main_result = self.classify_debiased(narration, main_q, main_labels)
        main_category = main_result["choice"]
        main_top3, top_ranked_main = self._top3(main_result["probabilities"])
        if main_category == others:
            sub_category = self.generate_dynamic_label(narration)
            top_ranked_sub = 0.0
            sub_top3 = "N/A (dynamically generated, not a closed-set probability)"
        else:
            sub_labels = self.taxonomy[main_category]
            sub_q = self.settings.sub_category_question.format(main_category=main_category)
            sub_result = self.classify_debiased(narration, sub_q, sub_labels)
            sub_category = sub_result["choice"]
            sub_top3, top_ranked_sub = self._top3(sub_result["probabilities"])
            # main_top3, top_ranked_main = self._top3(main_result["probabilities"])
        time_taken_sec = time.perf_counter() - start
        return {
            "main_category": main_category,
            "main_confidence": top_ranked_main,
            "main_top3": main_top3,
            "sub_category": sub_category,
            "sub_confidence": top_ranked_sub,
            "sub_top3": sub_top3,
            "time_taken_sec": round(time_taken_sec, 4),
        }


class LayaClassifier:
    """System-1 fast path via Convai's Laya. Non-autoregressive: every option is
    scored at its own [MASK] token and softmaxed within one forward pass, so
    it doesn't have the letter-position bias the causal-LM path needs
    classify_debiased() to correct for -- no shuffling/permutations needed.

    Same public method (classify_hierarchical) and same return shape as
    OpenJevClassifier, so it's a drop-in swap via settings.classifier_backend.
    """

    def __init__(self, taxonomy: dict, settings: AppSettings):
        import laya as laya_lib   # local import: only required if this backend is used
        self.taxonomy = taxonomy
        self.settings = settings
        self.agent = laya_lib.load(settings.laya_model_name)
        self._fallback_generator: "OpenJevClassifier | None" = None
        print(f"[LayaClassifier] model={settings.laya_model_name}")

    def _fallback(self) -> "OpenJevClassifier":
        # Laya is classification-only (no free-text generation). Reuse the
        # causal-LM path's short-generation trick ONLY for the 'Others' branch,
        # and only the first time it's actually needed -- so choosing the Laya
        # backend doesn't pay for a second model load unless a row lands in
        # 'Others'.
        if self._fallback_generator is None:
            self._fallback_generator = OpenJevClassifier(taxonomy=self.taxonomy, settings=self.settings)
        return self._fallback_generator

    @staticmethod
    def _top3_from_laya(answer: dict) -> str:
        # Field name for the per-option distribution isn't nailed down from the
        # model card alone -- try the likely keys, fall back to chosen+confidence
        # only. Print one raw result and adjust this if your installed version
        # names it differently.
        probs = answer.get("probabilities") or answer.get("distribution") or answer.get("scores")
        if not probs:
            return f"{answer.get('choice')} ({answer.get('confidence', 0):.1%})"
        ranked = sorted(probs.items(), key=lambda x: -x[1])[:3]
        top_conf = float(ranked[0][1])
        top3_distribution = " | ".join(f"{label} ({p:.1%})" for label, p in ranked)
        return top3_distribution, top_conf  #" | ".join(f"{label} ({p:.1%})" for label, p in ranked)

    def classify_hierarchical(self, narration: str) -> dict:
        start = time.perf_counter()
        others = self.settings.others_label

        main_q = {
            "main_category": {
                "type": "choice",
                "instructions": self.settings.main_category_question.format(others_label=others),
                "criteria": {label: label for label in self.taxonomy.keys()},
            }
        }
        main_result = self.agent.predict({"body": narration}, main_q)["answers"]["main_category"]
        main_category = main_result["choice"]
        main_top3, main_confidence = self._top3_from_laya(main_result)

        if main_category == others:
            sub_category = self._fallback().generate_dynamic_label(narration)
            sub_confidence = 0.0
            sub_top3 = "N/A (dynamically generated, not a closed-set probability)"
        else:
            sub_q = {
                "sub_category": {
                    "type": "choice",
                    "instructions": self.settings.sub_category_question.format(main_category=main_category),
                    "criteria": {label: label for label in self.taxonomy[main_category]},
                }
            }
            sub_result = self.agent.predict({"body": narration}, sub_q)["answers"]["sub_category"]
            sub_category = sub_result["choice"]
            # sub_top3 = self._top3_from_laya(sub_result)
            sub_top3, sub_confidence = self._top3_from_laya(sub_result)

        time_taken_sec = time.perf_counter() - start
        return {
            "main_category": main_category,
            "main_confidence": main_confidence,
            "main_top3": self._top3_from_laya(main_result),
            "sub_category": sub_category,
            "sub_confidence": sub_confidence,
            "sub_top3": sub_top3,
            "time_taken_sec": round(time_taken_sec, 4),
        }


def build_classifier(taxonomy: dict, settings: AppSettings):
    """Backend factory -- the only place that needs to know which System-1
    engine is active. Everything else (process_csv, CLI) just calls
    classify_hierarchical()."""
    if settings.classifier_backend == "laya":
        if settings.system1_model == settings.laya_model_name:
            raise ValueError(
                "settings.system1_model and settings.laya_model_name are both "
                f"'{settings.system1_model}'. The laya backend's 'Others' fallback "
                "(LayaClassifier._fallback) loads system1_model as a causal LM via "
                "AutoModelForCausalLM -- it needs a real causal checkpoint (e.g. a "
                "Qwen2.5 repo id), not Laya's own repo id. Fix system1_model in "
                "config.py (or whatever env var feeds it)."
            )
        return LayaClassifier(taxonomy=taxonomy, settings=settings)
    return OpenJevClassifier(taxonomy=taxonomy, settings=settings)


def process_csv(input_path: str, narration_col: str, output_path: str,
                 settings: AppSettings, regenerate_taxonomy: bool = False) -> None:
    df = pd.read_csv(input_path)
    if narration_col not in df.columns:
        raise ValueError(f"Column '{narration_col}' not found. Available columns: {list(df.columns)}")

    narrations = df[narration_col].astype(str).tolist()

    # --- System 2: design (or load cached) taxonomy, once per dataset ---
    taxonomy = load_or_generate_taxonomy(narrations, settings, force=regenerate_taxonomy)
    print(f"[taxonomy] {len(taxonomy)} main categories loaded from {settings.taxonomy_path}")

    # --- System 1: classify every row, no further LLM calls ---
    clf = build_classifier(taxonomy=taxonomy, settings=settings)
    rows = []
    for i, narration in enumerate(narrations, start=1):
        result = clf.classify_hierarchical(narration)
        rows.append(result)
        # print(f"[{i}/{len(df)}] > {narration[:50]}({result['main_confidence']:.1%}) > {result['main_category']} > {result['sub_category']}({result['sub_confidence']:.1%}) "
        #       f"({result['time_taken_sec']}s)")
        sub_conf = (
            f"({result['sub_confidence']:.1%})"
            if result.get("sub_confidence", 0.0) > 0.0
            else "(dynamic)"
        )
        print(
            f"[{i}/{len(df)}] > {narration[:50]} ({result['main_confidence']:.1%}) "
            f"> {result['main_category']} > {result['sub_category']} {sub_conf} "
            f"({result['time_taken_sec']}s)"
        )

    result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    result_df.to_csv(output_path, index=False)

    print(f"\nDone. {len(df)} rows -> {output_path}")
    print(f"Avg time/row : {result_df['time_taken_sec'].mean():.4f} sec")
    print(f"Total time   : {result_df['time_taken_sec'].sum():.2f} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid System-1/System-2 classifier for banking ERP cost narrations."
    )
    parser.add_argument("input_csv", help="Path to input CSV containing the narration column")
    parser.add_argument("--column", default="narration", help="Narration text column name (default: narration)")
    parser.add_argument("--output", default="classified_output.csv", help="Path to output CSV")
    parser.add_argument("--regenerate-taxonomy", action="store_true",
                         help="Force a fresh System-2 taxonomy design call even if taxonomy.json is cached")
    parser.add_argument("--backend", choices=["causal_lm", "laya"], default=None,
                         help="Override OPENJEV_CLASSIFIER_BACKEND for this run only "
                              "(causal_lm=Qwen2.5 logit read-off, laya=Convai Laya)")
    args = parser.parse_args()

    settings = default_settings
    if args.backend is not None:
        settings = default_settings.model_copy(update={"classifier_backend": args.backend})

    process_csv(args.input_csv, args.column, args.output, settings, args.regenerate_taxonomy)