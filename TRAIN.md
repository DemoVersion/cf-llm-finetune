# cf-llm-finetune

## Train
This project uses the `axolotl` library to fine-tune a smaller LLaMA 3.2 3B model on the synthetic parallel dataset generated from C++ ICPC solutions and their Python translations.
The configuration file for the fine-tuning is located in `config/llama-3.2-3b-lora.yml`.
Before starting the fine-tuning process, make sure to have the dataset generated and transformed as described in [DATASET.md](./DATASET.md).

### Fine-tuned model on Hugging Face
You can find the fine-tuned model on Hugging Face at [demoversion/Llama-3.2-3B-Instruct-PEFT-code-generation](https://huggingface.co/demoversion/Llama-3.2-3B-Instruct-PEFT-code-generation).


## Getting access to the base model
To access the base model, you need to log in to the Hugging Face Hub. You can do this by running the following command:
```bash
uv run huggingface-cli login
```
This will prompt you to enter your Hugging Face Hub credentials. After logging in, you will be able to access the base model specified in the configuration file. Remember that you request access to `meta-llama/Llama-3.2-3B-Instruct` before starting. For that fill out the form that they have at the hugging face website, they usally approve the request within 24 hours.

## Preprocess command to check the tokenized dataset
You can use the following command to preprocess the dataset and check the tokenized dataset:
```bash
uv run axolotl preprocess config/llama-3.2-3b-lora.yml
```

After this there will a folder named `last_run_prepared` in the current directory, which contains the tokenized dataset. you can load the tokenizer and dataset using the following code:
```python
from transformers import AutoTokenizer
from datasets import load_from_disk

ds = load_from_disk("./cf-llm-finetune/last_run_prepared/cbde416e6ffd6757d9a6228e7bd0e9f3")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
```
After that you can check the first item in the tokenized dataset like this:
```python
labels = ds['labels'][0]
tokens = [tok.decode([token_id]) for token_id in ds['input_ids'][0]]

print(list(zip(tokens,labels)))
```
This will print the tokens and their corresponding labels in the tokenized dataset. The output should look like this:
```
[('<|begin_of_text|>', -100),
 ('<|start_header_id|>', -100),
 ('system', -100),
 ('<|end_header_id|>', -100),
 ('\n\n', -100),
...
 ('<|eot_id|>', -100),
 ('<|start_header_id|>', -100),
 ('assistant', -100),
 ('<|end_header_id|>', -100),
 ('\n\n', -100),
 ('**', 334),
 ('Explanation', 70869),
...
]
```
Notice that the `-100` labels are used to ignore the tokens during the training process, and the other labels are the actual labels for the tokens. This is very important to understand that the trainer ignores the tokens with `-100` labels during the training process, because we don't want to train the model on system prompt and user input, and it should only focus on the assistant output.

To understand the prompt format better you can check the official [LLaMA 3.2 Model Cards & Prompt formats documentation](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/)

You can also get only the text part of the tokenized dataset, which is much more readable, using the following code:
```python
print(tok.decode(ds['input_ids'][0]))
```

## Train command
To start the fine-tuning process, you can use the following command:
```bash
uv run axolotl train config/llama-3.2-3b-lora.yml
```
To understand the configuration file better, you can check the [Axolotl Config Reference](https://docs.axolotl.ai/docs/config-reference.html) to change the parameters according to your needs.

Note that the `datasets.path` and `val_file` in the configuration file should point to the transformed dataset files that you generated and transformed according to [DATASET.md](./DATASET.md).

It's pointing to `./data/train_openai_response_transformed.jsonl` and `./data/val_openai_response_transformed.jsonl` by default.