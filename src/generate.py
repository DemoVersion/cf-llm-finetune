import requests
from joblib import Memory

from src.prompt import GENERATE_TEMPLATE, SYSTEM_PROMPT

memory = Memory("./cache", verbose=0)

OPENAI_CLIENT = None


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
