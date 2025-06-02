import os

import requests
from joblib import Memory
from openai import OpenAI

from src.prompt import GENERATE_TEMPLATE, SYSTEM_PROMPT

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

memory = Memory("./cache", verbose=0)


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
