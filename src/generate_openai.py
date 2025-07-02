from typing import Optional

import click
import openai
import pandas as pd
from tqdm import tqdm

from src.dataset import load_dataset_split
from src.generate import (
    call_openai_batch,
    generate_code,
    generate_messages,
    get_openai_batch_result,
)


@click.command()
@click.option(
    "--dataset-name",
    type=click.Choice(["train", "val", "test"], case_sensitive=False),
    default="val",
    help="Which dataset split to process: train, val, or test.",
)
@click.option(
    "--mode",
    type=click.Choice(["fast", "cheap"], case_sensitive=False),
    default="fast",
    help="Mode for generating responses: 'fast' for quick generation it will try calling directly and then switch to batch mode when rate limit exceeded, 'cheap' for cost-effective generation using batch processing.",
)
@click.option(
    "--batch-job-id",
    type=Optional[str],
    default=None,
    help="If provided, will use this batch job ID to fetch results instead of generating new responses. Only use this if you have already run a batch job and want to retrieve its results.",
)
def main(dataset_name, mode, batch_job_id):
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
    rate_limit_exceeded = False
    if mode == "cheap":
        rate_limit_exceeded = True
    batch_messages_list = []
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Processing {dataset_name} split"
    ):
        if rate_limit_exceeded is False:
            try:
                response = generate_code(row["source"], mode="openai")
            except openai.RateLimitError:
                rate_limit_exceeded = True
        if rate_limit_exceeded:
            batch_messages_list.append(generate_messages(row["source"]))
            continue
        responses.append(
            {
                "source": row["source"],
                "response": response,
                "contest_id": row.get("contest_id", None),
                "index": row.get("index", None),
            }
        )
    if len(batch_messages_list) > 0:
        click.echo(
            f"Rate limit exceeded, switching to batch processing for {len(batch_messages_list)} messages."
        )
        if batch_job_id is None:
            job_id = call_openai_batch(
                batch_messages_list,
            )
            click.echo(f"Batch job started with ID: {job_id}")
        else:
            job_id = batch_job_id
            click.echo(f"Using existing batch job ID: {job_id}")

        batch_results = get_openai_batch_result(job_id, poll_interval=30)
    result_df = pd.DataFrame(responses)
    output_file = f"{dataset_name}_openai_response.jsonl"
    result_df.to_json(output_file, orient="records", lines=True)
    click.echo(f"Wrote {len(responses)} responses to {output_file}")


if __name__ == "__main__":
    main()
