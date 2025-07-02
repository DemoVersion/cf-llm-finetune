import json
import os
import time
from tempfile import NamedTemporaryFile
from typing import Dict, List

import requests
from joblib import Memory
from openai import OpenAI

from src.prompt import GENERATE_TEMPLATE, SYSTEM_PROMPT

memory = Memory("./cache", verbose=0)

OPENAI_CLIENT = None


def call_openai_batch(
    messages_list: List[List[Dict]],
    model: str = "gpt-4.1",
    completion_window: str = "24h",
) -> str:
    """
    Create a Batch API job from multiple chat messages.
    Args:
        messages_list: A list of message sequences, each itself a list of dicts with 'role' and 'content'.
        model: The model to use for all batch requests.
        completion_window: How long the batch job may run (only "24h" is currently supported).
    Returns:
        The batch job ID as a string.
    """
    global OPENAI_CLIENT
    if OPENAI_CLIENT is None:
        OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_KEY"))

    with NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as tmp:
        for idx, messages in enumerate(messages_list):
            task = {
                "custom_id": f"batch-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": model, "messages": messages},
            }
            tmp.write(json.dumps(task) + "\n")
        tmp.flush()

        batch_file = OPENAI_CLIENT.files.create(
            file=open(tmp.name, "rb"), purpose="batch"
        )

    batch_job = OPENAI_CLIENT.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    return batch_job.id


def get_openai_batch_result(job_id: str, poll_interval: int = 30) -> List[Dict]:
    """
    Poll a batch job until completion, then download and parse its results.
    Args:
        job_id: The batch job ID returned by call_openai_batch.
        poll_interval: Seconds between status checks.
    Returns:
        A list of response objects, each containing 'custom_id' and 'response'.
    """
    global OPENAI_CLIENT
    if OPENAI_CLIENT is None:
        OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_KEY"))

    while True:
        job = OPENAI_CLIENT.batches.retrieve(id=job_id)
        status = job.status
        if status in ("succeeded", "failed"):
            break
        time.sleep(poll_interval)

    if status != "succeeded":
        raise RuntimeError(f"Batch job {job_id} ended with status {status}")

    output_id = job.output_file_id
    raw = OPENAI_CLIENT.files.content(output_id)
    lines = raw.content.decode("utf-8").splitlines()
    results = [json.loads(line) for line in lines]

    return results


@memory.cache
def call_openai_api(messages: list[dict], model: str = "gpt-4.1") -> str:
    """
    Call the OpenAI API with the provided messages and return the response.
    Args:
        messages (list[dict]): A list of message dictionaries to send to the API.
    Returns:
        str: The content of the response message.
    """
    global OPENAI_CLIENT
    if OPENAI_CLIENT is None:
        import os

        from openai import OpenAI

        OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    completion = OPENAI_CLIENT.chat.completions.create(
        model=model,
        store=True,
        messages=messages,
    )

    return completion.choices[0].message.content


@memory.cache
def call_api(messages: list[dict]) -> str:
    """
    Call the API with the provided messages and return the response.
    Args:
        messages (list[dict]): A list of message dictionaries to send to the API.
    Returns:
        str: The content of the response message.
    """

    url = "http://localhost:8080/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": messages,
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # Raise an error for bad responses
    response_json = response.json()
    return response_json["choices"][0]["message"]["content"]


def generate_messages(source_code: str) -> list[dict]:
    """
    Generate a list of messages for the model based on the source code.
    Args:
        source_code (str): The source code to generate messages for.
    Returns:
        list[dict]: A list of message dictionaries.
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": GENERATE_TEMPLATE.format(source_code=source_code),
        },
    ]
    return messages


def generate_code(source_code: str, mode: str = "local") -> str:
    """
    Generate code based on the provided prompt.
    Args:
        source_code (str): The source code to generate messages for.
    Returns:
        str: The generated code.
    """
    messages = generate_messages(source_code)
    if mode == "openai":
        response = call_openai_api(messages)
    elif mode == "transformers":
        response = generate_using_transformers(messages)
    else:
        response = call_api(messages)
    return response
