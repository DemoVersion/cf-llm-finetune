import click
import pandas as pd

from src.batch_generation import batch_process, generate_messages_dataset
from src.dataset import load_dataset_split
from src.model import ModelConfig, get_model
from src.utils import get_dict_key


def generate_cache_key(
    dataset_name,
    model_id,
    batch_size,
    lora_adapter,
    enable_quantization,
    output_path,
    use_cache,
):
    # Create a dictionary of the parameters
    params = {
        "dataset_name": dataset_name,
        "model_id": model_id,
        "batch_size": batch_size,
        "lora_adapter": lora_adapter,
        "enable_quantization": enable_quantization,
        "output_path": output_path,
        "use_cache": use_cache,
    }
    cache_key = get_dict_key(params)
    return cache_key


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
    "--model-id",
    default="meta-llama/Llama-3.2-3B-Instruct",
    help="Transformers model identifier to use.",
)
@click.option(
    "--batch-size", type=int, default=1, help="Batch size for processing messages."
)
@click.option(
    "--lora-adapter",
    type=str,
    default=None,
    help="Path to LoRA adapter to use with the model.",
)
@click.option(
    "--enable-quantization",
    is_flag=True,
    default=True,
    help="Enable quantization for the model.",
)
@click.option(
    "--output-path",
    type=click.Path(exists=False, writable=True, dir_okay=False),
    default="{dataset_name}_transformers_response.jsonl",
    help="Path to save the generated responses in JSONL format.",
)
@click.option(
    "--use-cache",
    is_flag=True,
    default=False,
    help="Use caching to skip previously processed messages. Useful for large datasets that run may terminate before completion.",
)
def generate_transformers(
    dataset_name,
    model_id,
    batch_size,
    lora_adapter,
    enable_quantization,
    output_path,
    use_cache,
):
    """
    Generate code responses using a Transformers model and save as JSONL.
    """
    train_df, val_df, test_df = load_dataset_split()
    datasets = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    if dataset_name == "functional-test":
        click.echo(
            "Using only the first 2 rows of the validation set for functional testing."
        )
        df = val_df.head(2)
    else:
        df = datasets[dataset_name]

    click.echo(f"Loaded {len(df)} rows from the {dataset_name} dataset.")
    if use_cache:
        click.echo("Using cache to skip previously processed messages.")
        cache_key = generate_cache_key(
            dataset_name,
            model_id,
            batch_size,
            lora_adapter,
            enable_quantization,
            output_path,
            use_cache,
        )
        click.echo(f"Cache key: {cache_key}")
        cache_file = f"cache_{cache_key}.pkl"
    else:
        click.echo("Not using cache, all messages will be processed.")
        cache_file = None
    messages = generate_messages_dataset(df)
    config = ModelConfig(
        enable_quantization=enable_quantization,
        lora_adapter=lora_adapter,
        # compile_kwargs={"fullgraph": True},
    )
    model = get_model(model_id=model_id, config=config)

    results = batch_process(
        messages,
        model,
        batch_size=batch_size,
        use_cache=use_cache,
        cache_file=cache_file,
    )

    responses = []
    for (_, row), resp in zip(df.iterrows(), results):
        responses.append(
            {
                "source": row["source"],
                "response": resp,
                "contest_id": row["contest_id"],
                "index": row["index"],
            }
        )

    result_df = pd.DataFrame(responses)
    if "{{dataset_name}}" in output_path:
        output_path_formatted = output_path.format(dataset_name=dataset_name)
    else:
        output_path_formatted = output_path
    click.echo(
        f"Saving responses to {output_path_formatted} with {len(result_df)} rows."
    )
    result_df.to_json(
        output_path_formatted, orient="records", lines=True, force_ascii=False
    )
    click.echo(f"Wrote {len(result_df)} responses to {output_path_formatted}")


if __name__ == "__main__":
    generate_transformers()
