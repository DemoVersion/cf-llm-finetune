import pandas as pd
from src.generate import generate_messages


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
