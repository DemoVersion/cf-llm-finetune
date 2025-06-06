import os

import requests
from joblib import Memory
from openai import OpenAI

from src.logger import logger
from src.prompt import GENERATE_TEMPLATE, SYSTEM_PROMPT

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

memory = Memory("./cache", verbose=0)

TRANSFORMERS_PIPES = dict()


def load_model(model_id: str = "meta-llama/Llama-3.2-3B-Instruct"):
    """
    Load the model for text generation.
    This function is called only once and caches the model for subsequent calls.
    """
    global TRANSFORMERS_PIPES
    if model_id not in TRANSFORMERS_PIPES:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        logger.info("Loading model for text generation...")

        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", load_in_4bit=True, torch_dtype=torch.float16
        )
        logger.info("Model loaded successfully.")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        TRANSFORMERS_PIPES[model_id] = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        logger.info("Transformers pipeline created successfully.")


def get_model(model_id: str = "meta-llama/Llama-3.2-3B-Instruct"):
    """
    Get the model for text generation.
    This function is called only once and caches the model for subsequent calls.
    Args:
        model_id (str): The model to use for text generation.
    Returns:
        transformers.pipeline: The transformers pipeline for text generation.
    """
    global TRANSFORMERS_PIPES
    if model_id not in TRANSFORMERS_PIPES:
        load_model(model_id)
    return TRANSFORMERS_PIPES[model_id]


@memory.cache
def generate_using_transformers(
    messages: list[dict], model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
) -> str:
    """
    Generate text using the transformers pipeline.
    Args:
        messages (list[dict]): A list of message dictionaries to send to the model.
        model_id (str): The model to use for text generation.
    Returns:
        str: The generated text from the model.
    """
    global TRANSFORMERS_PIPES
    if model_id not in TRANSFORMERS_PIPES:
        load_model(model_id)
    transformers_pipe = TRANSFORMERS_PIPES[model_id]
    result = transformers_pipe(
        messages,
        max_new_tokens=10000,
        do_sample=True,
        temperature=0.6,
    )
    return result[0]["generated_text"][-1]["content"]


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
