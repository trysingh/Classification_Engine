# Classification Engine

A dual-tier ("System 1" / "System 2") classification engine designed to hierarchically categorize any dataset into multiple level. The current project is doing at 2 Levels. Taxonomy can be generated at any level

---

## 🏛 Architecture Overview

The system uses a hybrid approach balancing latency, cost, and classification accuracy:

- **System 2 (Slow Path / Taxonomy Designer):**
  - Powered by an LLM via **Ollama** (default: `gemma4:31b-cloud`).
  - Samples ERP narrations and synthesizes a structured hierarchical taxonomy (`Main Category` $\rightarrow$ `Sub-category`) saved to `taxonomy.json`.
  - Enforces domain constraints (e.g., maximum main/sub-categories, and a dedicated dynamic `"Others"` catch-all category).

- **System 1 (Fast Path / Per-Row Classifier):**
  - Runs on every ledger transaction row for low-latency inference.
  - Supports two interchangeable inference backends:
    1. **`causal_lm` (Default):** Runs local logit read-off using small instruction-tuned models (e.g., `Qwen/Qwen2.5-0.5B-Instruct`). Employs **permutation debiasing** (averaging multiple shuffled option orderings) to mitigate positional bias in multiple-choice logit scoring.
    2. **`laya`:** Uses Convai's `convaiinnovations/laya` non-autoregressive option-marker scoring (~33ms/call), structurally immune to positional bias. It falls back to `system1_model` when generating open-ended labels for the `"Others"` category branch.

---

## ⚙️ Configuration & Environment Variables

All settings are managed via Pydantic in `config.py`. You can override any value using an environment variable prefixed with **`OPENJEV_`** or by specifying them in a `.env` file.

### Key Configuration Parameters

| Environment Variable | Config Field | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `OPENJEV_CLASSIFIER_BACKEND` | `classifier_backend` | `causal_lm` | Fast path engine: `causal_lm` or `laya`. |
| `OPENJEV_SYSTEM1_MODEL` | `system1_model` | `Qwen/Qwen2.5-0.5B-Instruct` | Local Causal LM checkpoint for fast-path inference and fallback generation. |
| `OPENJEV_LAYA_MODEL_NAME` | `laya_model_name` | `convaiinnovations/laya` | Model identifier for Convai Laya backend. |
| `OPENJEV_OLLAMA_BASE_URL` | `ollama_base_url` | `http://localhost:11434` | Endpoint for the Ollama instance running System 2. |
| `OPENJEV_SYSTEM2_MODEL` | `system2_model` | `gemma4:31b-cloud` | LLM model for taxonomy generation. |
| `OPENJEV_SYSTEM2_TIMEOUT_SEC`| `system2_timeout_sec` | `120` | Request timeout for Ollama (in seconds). |
| `OPENJEV_SYSTEM2_SEED` | `system2_seed` | `42` | Seed for reproducible taxonomy creation (`None` to randomize). |
| `OPENJEV_MAX_MAIN_CATEGORIES`| `max_main_categories` | `7` | Maximum allowable main categories in taxonomy. |
| `OPENJEV_MAX_SUB_CATEGORIES` | `max_sub_categories` | `10` | Maximum sub-categories per main category. |
| `OPENJEV_TAXONOMY_SAMPLE_SIZE`| `taxonomy_sample_size`| `40` | Number of ledger samples provided to System 2 for clustering. |
| `OPENJEV_TAXONOMY_PATH` | `taxonomy_path` | `taxonomy.json` | Destination path for the generated taxonomy JSON. |
| `OPENJEV_OTHERS_LABEL` | `others_label` | `Others` | Name of the fallback / catch-all category. |
| `OPENJEV_DEBIAS_PERMUTATIONS`| `debias_permutations` | `3` | Shuffled forward passes run to cancel option position bias. |

> **Important Constraint:** `system1_model` cannot be set to the same repository as `laya_model_name`. When running the `laya` backend, `system1_model` is required as a causal fallback to generate open-ended labels for the `"Others"` category branch.

---

## 🚀 Quickstart

### 1. Prerequisites

- Python 3.10+
- An accessible [Ollama](https://ollama.ai/) instance running the configured System 2 model:
  ```bash
  ollama pull gemma4:31b-cloud
  ```

### 2. Environment Setup

Create a `.env` file in the root directory if you want to override defaults:

```dotenv
# .env
OPENJEV_CLASSIFIER_BACKEND=causal_lm
OPENJEV_SYSTEM1_MODEL=Qwen/Qwen2.5-0.5B-Instruct
OPENJEV_OLLAMA_BASE_URL=http://localhost:11434
OPENJEV_DEBIAS_PERMUTATIONS=3
```

### 3. Execution Flow

1. **Taxonomy Bootstrapping (System 2):**
   - Samples $N$ (`taxonomy_sample_size`) rows from the ledger narrations.
   - Prompts the Ollama model to generate a structured schema adhering to the category limits.
   - Saves output to `taxonomy.json`.

2. **Hierarchical Classification (System 1):**
   - **Step 1 (Main Category):** Scores narration against main categories using selected `classifier_backend`.
   - **Step 2 (Sub-Category):**
     - If mapped to a standard main category, classifies into its respective sub-categories.
     - If mapped to `"Others"`, prompts the causal fallback (`system1_model`) with `dynamic_label_prompt` to synthesize an open-ended 2–4 word sub-category.

---

## 🗂 Taxonomy File Format

The system expects and produces a JSON taxonomy structured as:

```json
{
  "taxonomy": {
    "Infrastructure": ["Cloud Compute", "Storage", "Data Center"],
    "Software Licensing": ["Core Banking", "SaaS Subscriptions"],
    "Maintenance & Support": ["Vendor SLA", "Annual Maintenance"],
    "Others": []
  }
}
```
*(Note: The `"Others"` branch must always contain an empty list as sub-categories are dynamically inferred).*


>
### When calling Laya
python Classify.py source_file.csv --backend laya
python Classify.py "data\samples.csv" --output "data/samples_output.csv" --backend laya --regenerate-taxonomy

### When calling other models like qwen etc
python Classify.py source_file.csv --backend causal_lm
python Classify.py "data\samples.csv" --output "data/samples_output.csv" --backend casual_lm --regenerate-taxonomy

