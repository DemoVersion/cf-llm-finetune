# cf-llm-finetune

## Overview

`cf-llm-finetune` is a specialized project for converting C++ ICPC solutions from Codeforces into Python code, and verifying the correctness of each translation by running it against sample inputs. We generate a synthetic parallel dataset using codeforces datasets provided by DeepSeek, specifically:
- `"open-r1/codeforces-submissions"`
- `"open-r1/codeforces"`

Synthetic Python translations are produced via GPT-4.1, then used to fine-tune a smaller LLaMA 3.2 3B model. Our goal is to explore how effectively lightweight models can be trained for this focused code‐translation task.

### Generated Dataset and Fine-Tuned Model on Hugging Face
The generated dataset is available on Hugging Face at [demoversion/cf-cpp-to-python-code-generation](https://huggingface.co/datasets/demoversion/cf-cpp-to-python-code-generation).  
The fine-tuned model can be found at [demoversion/Llama-3.2-3B-Instruct-PEFT-code-generation](https://huggingface.co/demoversion/Llama-3.2-3B-Instruct-PEFT-code-generation). If you do not want to modify the dataset generation process, you can use this dataset directly for fine-tuning. Or if you only want to do inference, you can use the pre-trained model directly without fine-tuning.

## Dataset

This project uses a synthetic parallel dataset built from the [Codeforces submissions](https://huggingface.co/datasets/open-r1/-submissions) and [problems](https://huggingface.co/datasets/open-r1/codeforces). C++ ICPC-style solutions are filtered, cleaned, and paired with problem statements to generate Python translations using GPT-4.1, creating a fine-tuning dataset for code translation.

The final dataset consists of C++ solutions from 2,000 unique problems, and synthetic Python answers, split into train (1,400), validation (300), and test (300) sets. For details on dataset generation, cleaning, evaluation, and translation process, see [DATASET.md](./DATASET.md).

Before running the dataset generation, ensure that you have completed the installation steps outlined below.

## Model Fine-Tuning

- **Base model:** LLaMA 3.2 3B Instruct
- **Framework:** `axolotl`, `transformers`, `deepspeed`, `flash-attn`  

This project fine-tunes the `meta-llama/Llama-3.2-3B-Instruct` model using the `Axolotl` library and a synthetic C++ to Python dataset. Training uses a LoRA configuration defined in `config/llama-3.2-3b-lora.yml`.

Before training, ensure:
- You've completed the installation steps outlined below.
- You've generated the dataset (see [DATASET.md](./DATASET.md)).
- You've been granted access to the base model on Hugging Face.
- You’ve logged in via huggingface-cli login.

You can inspect the tokenized dataset with:
```bash
uv run axolotl preprocess config/llama-3.2-3b-lora.yml
```
To start training:
```bash
uv run axolotl train config/llama-3.2-3b-lora.yml
```
More details, including tokenizer inspection and prompt formatting, can be found in [TRAIN.md](./TRAIN.md).

## Inference and Evaluation
After instalation and training, or by using the pre-trained model, you can run inference on the fine-tuned model using the following command:
```bash
uv run python -m src.generate_transformers --dataset-name test --lora-adapter demoversion/Llama-3.2-3B-Instruct-PEFT-code-generation --output-path test_finetunned_transformers_response.jsonl --use-cache
```
For inference of the base model, you can use:
```bash
uv run python -m src.generate_transformers --dataset-name test --output-path test_transformers_response.jsonl --use-cache
```

After that for evaluation, you can run the following command:
```bash
uv run python -m src.evaluate_response --response-file test_finetunned_transformers_response.jsonl
```

If you have fine-tuned the model locally, for generation you can use the relative path to the LoRA adapter:
```bash
uv run python -m src.generate_transformers --dataset-name test --lora-adapter ./outputs/cf-llm-finetune-llama-3.2-3b-lora --output-path test_finetunned_transformers_response.jsonl --use-cache
```


## Installation

1. Install `uv` (if not already installed):  
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
uv sync --extra dev --extra gpu
```

5. (Optional) Enable Flash Attention

To speed up training, add `flash-attn` to the `pyproject.toml` and rerun the sync step.


