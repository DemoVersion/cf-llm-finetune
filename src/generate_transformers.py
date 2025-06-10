import click
import pandas as pd

from src.batch_generation import batch_process, generate_messages_dataset
from src.dataset import load_dataset_split
from src.model import ModelConfig, get_model


@click.command()
@click.option(
    "--dataset-name",
    type=click.Choice(["train", "val", "test"], case_sensitive=False),
    default="val",
    help="Which dataset split to process: train, val, or test.",
)
@click.option(
    "--model-id",
    default="meta-llama/Llama-3.2-3B-Instruct",
    help="Transformers model identifier to use.",
)
@click.option(
    "--batch-size", type=int, default=1, help="Batch size for processing messages."
)
def generate_transformers(dataset_name, model_id, batch_size):
    """
    Generate code responses using a Transformers model and save as JSONL.
    """
    train_df, val_df, test_df = load_dataset_split()
    datasets = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    df = datasets[dataset_name]

    messages = generate_messages_dataset(df)
    config = ModelConfig(compile_kwargs={"fullgraph": True})
    model = get_model(model_id=model_id, config=config)

    results = batch_process(messages, model, batch_size=batch_size)

    responses = []
    for row, resp in zip(df.itertuples(index=False), results):
        responses.append(
            {
                "source": getattr(row, "source", None),
                "response": resp,
                "contest_id": getattr(row, "contest_id", None),
                "index": getattr(row, "index", None),
            }
        )

    result_df = pd.DataFrame(responses)
    output_file = f"{dataset_name}_transformers_response.jsonl"
    result_df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    click.echo(f"Wrote {len(result_df)} responses to {output_file}")


if __name__ == "__main__":
    generate_transformers()
