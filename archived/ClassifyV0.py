"""
open_jev_zero_shot.py
----------------------
Open-source re-implementation of the "System 1 / Jev" decision pattern:
instead of letting an LLM generate free text, read the next-token
probabilities for a fixed set of candidate labels directly off the
model's logits, in a single forward pass. Same trick used by community
projects like TheoLeeCJ/openjev and razorback16/openjev.

Requires:
    pip install torch transformers --break-system-packages

Swap MODEL_NAME for any small causal LM you have local/HF access to
(Qwen2.5-0.5B-Instruct, Gemma-3-270M, gpt2, etc). Smaller = faster,
closer to the ~15-50ms figures reported for edge-sized OpenJev builds.
"""

import time
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Chart of cost accounts, refined from observed "Others" fallout on real data.
# 7 main categories, each capped at <=10 sub-categories.
TAXONOMY = {
    "Infrastructure & Hosting": [
        "Data Center Costs",
        "Cloud Compute (IaaS/PaaS)",
        "Storage & Backup",
        "Disaster Recovery",
        "Compute / Memory Provisioning",
        "Server Hardware",
    ],
    "Network & Connectivity": [
        "Internet & Telecommunications",
        "WAN / MPLS Links",
        "Bandwidth & Data Transfer",
        "VPN & Remote Access",
    ],
    "Governance, Risk & Compliance": [
        "Regulatory / Audit Compliance",
        "Security Tooling",
        "Penetration Testing",
        "Data Privacy & Governance Tools",
        "Policy & Controls Management",
    ],
    "Software Licensing": [
        "New License Purchase",
        "License Renewal",
        "License Upgrade / Add-on",
        "SaaS Subscription Seats",
        "API / Usage-based Licensing",
        "Server / OS Licenses",
        "BI & Analytics Tool Licenses",
    ],
    "Operations": [
        "Cost of Operations (Run & Manage)",
        "IT Service Management",
        "Application Monitoring & Observability",
        "Batch / Job Processing",
        "Incident & Problem Management",
    ],
    "Consulting & Professional Services": [
        "Implementation & Consulting",
        "Customization / Development",
        "System Integration",
        "Software Development Costs",
        "Training & Certification",
    ],
    "Maintenance & Support": [
        "Annual Maintenance Contract (AMC)",
        "Vendor Support Retainer",
        "Patch / Version Update",
        "Enterprise Support Plans",
    ],
}


class OpenJevClassifier:
    """Zero-shot 'Choice' primitive: one forward pass -> label probabilities."""

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        self.model.eval()
        print(f"[OpenJevClassifier] model={model_name} device={self.device} dtype={dtype}")

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

        # Candidate tokens = first-token id of each option letter (A, B, C, ...)
        letters = [chr(65 + i) for i in range(len(labels))]
        candidate_ids = [self.tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]

        start = time.perf_counter()
        logits = self.model(**inputs).logits[0, -1]        # single forward pass, no generation
        latency_ms = (time.perf_counter() - start) * 1000

        candidate_logits = logits[candidate_ids]
        probs = torch.softmax(candidate_logits, dim=-1)

        return {
            "prompt_tokens": inputs["input_ids"].shape[-1],
            "latency_ms": round(latency_ms, 2),
            "choice": labels[int(probs.argmax())],
            "probabilities": {
                labels[i]: round(float(probs[i]), 4) for i in range(len(labels))
            },
        }

    @staticmethod
    def _top3(probabilities: dict) -> str:
        ranked = sorted(probabilities.items(), key=lambda x: -x[1])[:3]
        return " | ".join(f"{label} ({p:.1%})" for label, p in ranked)

    @torch.inference_mode()
    def generate_dynamic_label(self, narration: str, max_new_tokens: int = 6) -> str:
        """Fallback for the 'Others' branch: Choice can't produce an open-ended
        label, so this does a short greedy generation (a few tokens, not a full
        response) to pull out a specific sub-category phrase instead of the
        static word 'Others'."""
        prompt = (
            f"ERP cost narration: {narration}\n"
            f"In 2-4 words, name the most specific cost sub-category this belongs to:\n"
            f"Sub-category:"
        )
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
        """Two-stage Choice: Main category, then Sub-category within it.
        Two forward passes -> two small, targeted decisions rather than
        one flat choice over every leaf label."""
        start = time.perf_counter()

        main_q = ("What is the main cost category of this ERP narration for a banking "
                   "software system? Choose 'Others' only if none of the categories fit.")
        main_labels = list(TAXONOMY.keys()) + ["Others"]
        main_result = self.classify(narration, main_q, main_labels)
        main_category = main_result["choice"]

        if main_category == "Others":
            sub_category = self.generate_dynamic_label(narration)
            sub_top3 = "N/A (dynamically generated, not a closed-set probability)"
        else:
            sub_labels = TAXONOMY[main_category]
            sub_q = f"Within '{main_category}', what is the specific sub-category of this ERP narration?"
            sub_result = self.classify(narration, sub_q, sub_labels)
            sub_category = sub_result["choice"]
            sub_top3 = self._top3(sub_result["probabilities"])

        time_taken_sec = time.perf_counter() - start

        return {
            "main_category": main_category,
            "main_top3": self._top3(main_result["probabilities"]),
            "sub_category": sub_category,
            "sub_top3": sub_top3,
            "time_taken_sec": round(time_taken_sec, 4),
        }


def process_csv(input_path: str, narration_col: str, output_path: str, model_name: str = MODEL_NAME) -> None:
    df = pd.read_csv(input_path)
    if narration_col not in df.columns:
        raise ValueError(f"Column '{narration_col}' not found. Available columns: {list(df.columns)}")

    clf = OpenJevClassifier(model_name=model_name)

    rows = []
    for i, narration in enumerate(df[narration_col].astype(str), start=1):
        result = clf.classify_hierarchical(narration)
        rows.append(result)
        print(f"[{i}/{len(df)}] {result['main_category']} > {result['sub_category']} "
              f"({result['time_taken_sec']}s)")

    result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    result_df.to_csv(output_path, index=False)

    print(f"\nDone. {len(df)} rows -> {output_path}")
    print(f"Avg time/row : {result_df['time_taken_sec'].mean():.4f} sec")
    print(f"Total time   : {result_df['time_taken_sec'].sum():.2f} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify banking ERP cost narrations into Main/Sub category (OpenJev-style zero-shot)."
    )
    parser.add_argument("input_csv", help="Path to input CSV containing the narration column")
    parser.add_argument("--column", default="Narration", help="Narration text column name (default: narration)")
    parser.add_argument("--output", default="classified_output.csv", help="Path to output CSV")
    parser.add_argument("--model", default=MODEL_NAME, help="HF causal LM name/path")
    args = parser.parse_args()

    process_csv(args.input_csv, args.column, args.output, args.model)