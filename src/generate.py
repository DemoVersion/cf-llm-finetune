import requests
from src.prompt import SYSTEM_PROMPT, GENERATE_TEMPLATE


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
        "model": "your-model-name",  # Replace with your model's name
        "messages": [messages],
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()[0]["message"]["content"]


def generate_code(source_code: str) -> str:
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

    response = call_api(messages)
    return response
