from typing import Dict, Optional

import click
import openai
import pandas as pd
from tqdm import tqdm
from traitlets import List

from src.dataset import load_dataset_split
from src.generate import (
    call_openai_batch,
    generate_code,
    generate_messages,
    get_openai_batch_result,
)
from src.utils import get_dict_key


def create_messages_key_to_result_map(
    batch_results: List[Dict], input_file_rows: List[Dict]
) -> Dict[str, str]:
    """
    Create a mapping from messages key to result from the batch results.
    """
    batch_id_to_messages_key = dict()
    for row in input_file_rows:
        key = get_dict_key(row["body"]["messages"])
        batch_id_to_messages_key[row["custom_id"]] = key

    messages_key_to_result = dict()
    for row in batch_results:
        input_messages_key = batch_id_to_messages_key[row["custom_id"]]
        result = row["response"]["body"]["choices"][0]["message"]["content"]
        messages_key_to_result[input_messages_key] = result
    return messages_key_to_result


@click.command()
@click.option(
    "--dataset-name",
    type=click.Choice(
        ["train", "val", "test", "functional-test"], case_sensitive=False
    ),
    default="val",
    help="Which dataset split to process: train, val, or test or functional-test for quick validation.",
)
@click.option(
    "--mode",
    type=click.Choice(["fast", "cheap"], case_sensitive=False),
    default="fast",
    help="Mode for generating responses: 'fast' for quick generation it will try calling directly and then switch to batch mode when rate limit exceeded, 'cheap' for cost-effective generation using batch processing.",
)
@click.option(
    "--batch-job-id",
    type=str,
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
    if dataset_name == "functional-test":
        click.echo(
            "Using only the first 10 rows of the validation set for functional testing."
        )
        df = val_df.head(10)
    else:
        df = datasets[dataset_name]

    responses = []
    rate_limit_exceeded = False
    if mode == "cheap":
        rate_limit_exceeded = True
    batch_messages_list = []
    processed_sources = set()
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
        processed_sources.add(row["source"])

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

        batch_results, batch_input_rows = get_openai_batch_result(
            job_id, poll_interval=30
        )
        click.echo(f"Batch job {job_id} completed, processing results...")
        messages_key_to_result = create_messages_key_to_result_map(
            batch_results, batch_input_rows
        )

        for _, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Processing {dataset_name} split"
        ):
            if row["source"] in processed_sources:
                continue
            input_messages_key = get_dict_key(generate_messages(row["source"]))
            response = messages_key_to_result.get(input_messages_key, None)
            if response:
                responses.append(
                    {
                        "source": row["source"],
                        "response": response,
                        "contest_id": row.get("contest_id", None),
                        "index": row.get("index", None),
                    }
                )
            else:
                click.echo(
                    f"Warning: No response found for source: {row['source']} with key: {input_messages_key}"
                )

    result_df = pd.DataFrame(responses)
    output_file = f"{dataset_name}_openai_response.jsonl"
    result_df.to_json(output_file, orient="records", lines=True)
    click.echo(f"Wrote {len(responses)} responses to {output_file}")


if __name__ == "__main__":
    main()
