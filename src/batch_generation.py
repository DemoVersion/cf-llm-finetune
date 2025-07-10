import os
import pickle
from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm

from src.generate import generate_messages

cache = {}


def generate_from_token_ids(
    model, tokenizer, tokenized_inputs, max_new_tokens, temperature
):
    with torch.no_grad():
        return model.generate(
            **tokenized_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
        )


def decode_token_ids_to_messages(tokenizer, generated_ids, tokenized_inputs):
    # get only the generated IDs after the input
    input_length = tokenized_inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_length:]
    all_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    outputs = []
    for i, text in enumerate(all_outputs):
        outputs.append(text.strip())
    return outputs


def process_all_conversations(
    model, tokenizer, conversations, max_new_tokens=4096, temperature=0.6
):
    # Each conversation is a list of messages
    input_texts = [
        tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        for messages in conversations
    ]
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized_inputs_org = tokenizer(
        input_texts, padding="longest", return_tensors="pt"
    )
    tokenized_inputs = {k: v.to(model.device) for k, v in tokenized_inputs_org.items()}

    generated_ids = generate_from_token_ids(
        model,
        tokenizer,
        tokenized_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return (
        decode_token_ids_to_messages(tokenizer, generated_ids, tokenized_inputs_org),
        tokenized_inputs_org,
        generated_ids,
    )


def load_cache(cache_file):
    """
    Load the cache dictionary from disk into the global `cache` variable.
    """
    global cache
    if cache_file is None:
        return
    if len(cache) == 0 and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)


def save_cache(cache_file):
    """
    Persist the global `cache` dictionary to disk.
    """
    if cache_file is None:
        return
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)


def process_chunk(
    chunk,
    model,
    tokenizer,
    max_new_tokens=4096,
    temperature=0.6,
    use_cache=False,
    cache_file=None,
):
    """
    Process a list of messages (chunk). Uses per-message caching:
    - Checks each message against the cache.
    - If any are uncached, calls process_all_conversations once on the list of uncached messages.
    - Caches results for each message.
    - Returns a list of (outputs, tokenized_input, generated_ids) tuples in chunk order.

    Args:
        chunk (list): A list of message dicts.
        model: The chat model.
        tokenizer: The tokenizer associated with the model.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.

    Returns:
        list: List of tuples for each message in chunk.
    """
    load_cache(cache_file)

    # Identify which messages need processing
    uncached = []
    for msg in chunk:
        key = repr(msg)
        if key not in cache or not use_cache:
            uncached.append(msg)

    # Process uncached messages in one batch call
    if uncached:
        print(f"[{datetime.now()}] Processing {len(uncached)} new messages in chunk.")
        outputs_new, tokenized_input_new, generated_ids_new = process_all_conversations(
            model,
            tokenizer,
            uncached,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        print("Size of tokenized input:", tokenized_input_new["input_ids"].shape[1])
        print("Size of generated IDs:", generated_ids_new.shape[1])
        for i, msg in enumerate(uncached):
            key = repr(msg)
            cache[key] = (
                outputs_new[i],
                (
                    tokenized_input_new[i]
                    if isinstance(tokenized_input_new, list)
                    else tokenized_input_new
                ),
                (
                    generated_ids_new[i]
                    if isinstance(generated_ids_new, list)
                    else generated_ids_new
                ),
            )
        save_cache(cache_file)
    else:
        print(
            f"[{datetime.now()}] All messages in chunk are cached. Skipping processing."
        )

    # Retrieve results in original chunk order
    results = []
    for msg in chunk:
        key = repr(msg)
        results.append(cache[key][0])  # Get only the output text

    return results


def batch_process(all_messages, model, batch_size=5, cache_file=None, **kwargs):
    """
    Iterate through all_messages in batches of `batch_size`, using caching to skip
    previously processed chunks.

    Args:
        all_messages (list): Full list of message dicts.
        model: The chat model.
        tokenizer: The tokenizer.
        batch_size (int): Number of messages per chunk.
        **kwargs: Passed to process_chunk.

    Returns:
        list: A list of results for each processed chunk.
    """
    load_cache(cache_file)
    results = []

    for i in tqdm(range(0, len(all_messages), batch_size)):
        chunk = all_messages[i : i + batch_size]
        outputs = process_chunk(
            chunk, model.model, model.tokenizer, cache_file=cache_file, **kwargs
        )

        results.extend(outputs)

    return results


def generate_messages_dataset(dataset: pd.DataFrame) -> list[list[dict]]:
    """
    Generate messages for each row in the dataset.

    Args:
        dataset (pd.DataFrame): The dataset containing the source code and examples.
        model_id (str): The model to use for text generation.

    Returns:
        list[list[dict]]: A list of lists of messages for each row in the dataset.
    """
    messages_list = []
    for _, row in dataset.iterrows():
        messages = generate_messages(row["source"])
        messages_list.append(messages)
    return messages_list
