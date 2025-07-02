import click
import openai
import pandas as pd
from tqdm import tqdm

from src.dataset import load_dataset_split
from src.generate import generate_code


@click.command()
@click.option(
    "--dataset-name",
    type=click.Choice(["train", "val", "test"], case_sensitive=False),
    default="val",
    help="Which dataset split to process: train, val, or test.",
)
def main(dataset_name):
    """
    Generate code responses with openai for the specified dataset split and save them in JSONL format.
    """
    train_df, val_df, test_df = load_dataset_split()
    datasets = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    df = datasets[dataset_name]

    responses = []

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Processing {dataset_name} split"
    ):
        try:
            response = generate_code(row["source"], mode="openai")
        except openai.RateLimitError:
            click.echo("Rate limit exceeded. Please try again later.")
            break
        responses.append(
            {
                "source": row["source"],
                "response": response,
                "contest_id": row.get("contest_id", None),
                "index": row.get("index", None),
            }
        )

    result_df = pd.DataFrame(responses)
    output_file = f"{dataset_name}_openai_response.jsonl"
    result_df.to_json(output_file, orient="records", lines=True)
    click.echo(f"Wrote {len(responses)} responses to {output_file}")


if __name__ == "__main__":
    main()
