import os

import requests
from joblib import Memory
from openai import OpenAI

from src.prompt import GENERATE_TEMPLATE, SYSTEM_PROMPT

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

memory = Memory("./cache", verbose=0)

TRANSFORMERS_PIPE = None


@memory.cache
def generate_using_transformers(
    messages: list[dict], model: str = "meta-llama/Llama-3.2-3B-Instruct"
) -> str:
    """
    Generate text using the transformers pipeline.
    Args:
        messages (list[dict]): A list of message dictionaries to send to the model.
        model (str): The model to use for text generation.
    Returns:
        str: The generated text from the model.
    """

    global TRANSFORMERS_PIPE
    if TRANSFORMERS_PIPE is None:
        from transformers import pipeline

        TRANSFORMERS_PIPE = pipeline(
            "text-generation",
            model=model,
            load_in_8bit=True,
            device_map="cuda:0",
        )
    result = TRANSFORMERS_PIPE(messages)
    return result["generated_text"][-1]["content"]


@memory.cache
def call_openai_api(messages: list[dict], model: str = "gpt-4.1") -> str:
    """
    Call the OpenAI API with the provided messages and return the response.
    Args:
        messages (list[dict]): A list of message dictionaries to send to the API.
    Returns:
        str: The content of the response message.
    """
    completion = client.chat.completions.create(
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


def generate_code(source_code: str, mode: str = "local") -> str:
    """
    Generate code based on the provided prompt.
    Args:
        prompt (str): The prompt to generate code for.
    Returns:
        str: The generated code.
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
    if mode == "openai":
        response = call_openai_api(messages)
    else:
        response = call_api(messages)
    return response
