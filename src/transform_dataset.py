import click
import pandas as pd
from tqdm import tqdm

from src.generate import (
    generate_messages,
)


@click.command()
@click.option(
    "--dataset-path",
    "dataset_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the dataset file in JSON format.",
    required=True,
)
def transform(dataset_path: str):
    """
    Generate responses for the specified dataset file using OpenAI API. The response format is in Chat Template format and JSON Lines format.
    """
    df = pd.read_json(dataset_path, orient="records", lines=True)
    click.echo(f"Loaded dataset with {len(df)} rows from {dataset_path}")
    transformed_arr = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_path}"):
        messages = generate_messages(row["source"])
        messages.append(
            {
                "role": "assistant",
                "content": row["response"],
            }
        )
        transformed_arr.append(
            {
                "messages": messages,
                "source": row["source"],
                "contest_id": row.get("contest_id", None),
                "index": row.get("index", None),
            }
        )
    target_df_path = dataset_path.replace(".jsonl", "_transformed.jsonl")
    if target_df_path == dataset_path:
        target_df_path = dataset_path + "_transformed.jsonl"
    result_df = pd.DataFrame(transformed_arr)
    result_df.to_json(target_df_path, orient="records", lines=True)
    click.echo(f"Wrote transformed dataset to {target_df_path}")


if __name__ == "__main__":
    transform()
