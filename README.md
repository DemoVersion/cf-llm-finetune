# cf-llm-finetune

## Overview

`cf-llm-finetune` is a specialized project for converting C++ ICPC solutions from Codeforces into Python code, and verifying the correctness of each translation by running it against sample inputs. We generate a synthetic parallel dataset using codeforces datasets provided by DeepSeek, specifically:
- `"open-r1/codeforces-submissions"`
- `"open-r1/codeforces"`

Synthetic Python translations are produced via GPT-4.1, then used to fine-tune a smaller LLaMA 3.2 3B model. Our goal is to explore how effectively lightweight models can be trained for this focused code‐translation task.

## Dataset

1. **Source C++ solutions**  
   - Codeforces submissions labeled “ICPC” difficulty  
   - Pulled via the `open-r1/codeforces-submissions` dataset

2. **Problem statements & samples**  
   - Retrieved from `open-r1/codeforces`

3. **Synthetic translations**  
   - Generated with GPT-4.1  
   - Paired with original C++ code to form a fine-tuning corpus

## Model Fine-Tuning

- **Base model:** LLaMA 3.2 3B  
- **Framework:** `uv`  
- **Objective:** Minimize translation errors and ensure runtime correctness on sample tests  
- **Verification:** Automated test harness that compiles/runs the Python output against provided samples

## Installation

1. **Install `uv`** (if not already installed):  
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

2. Configure CUDA Backend

Select the matching `UV_TORCH_BACKEND` for your CUDA version (prebuilt wheels may not exist for every combination). For CUDA 11.8:

```bash
export UV_TORCH_BACKEND=cu118
```

3. Install Build Tools

```bash
uv pip install packaging setuptools wheel
```

4. Sync Development Dependencies

```bash
uv sync --dev
```

5. (Optional) Enable Flash Attention

To speed up training, add `flash-attn` to the `pyproject.toml` and rerun the sync step.


