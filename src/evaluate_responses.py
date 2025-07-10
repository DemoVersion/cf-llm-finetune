import click
import pandas as pd
from tqdm import tqdm

from src.code_runner import run_script
from src.dataset import load_experiment_dataset
from src.postprocess import postprocess_response
from src.utils import judge_output


def evaluate_responses(response_file, log_level="INFO"):
    exp_df = load_experiment_dataset()  # must contain contest_id, index, examples
    resp_df = pd.read_json(response_file, orient="records", lines=True)
    merged = pd.merge(
        resp_df, exp_df, on=["contest_id", "index"], how="inner", suffixes=("", "_exp")
    )

    click.echo(f"Total merged submissions: {len(merged)}")

    passed, failed = 0, 0
    for i, row in tqdm(
        merged.iterrows(), total=len(merged), desc="Evaluating submissions"
    ):
        code = postprocess_response(row["response"])
        if log_level == "DEBUG":
            click.echo(f"\nSubmission {i} code:\n{code}\n")

        submission_failed = False
        examples = row.get("examples") or []
        if not isinstance(examples, list) or len(examples) == 0:
            if log_level == "DEBUG":
                click.echo(f"Skipping submission {i} (no examples)")
            continue
        for test in examples:
            inp = test["input"]
            expected = test["output"]
            if log_level == "DEBUG":
                click.echo(f"Running test: {inp}")
            result, stderr = run_script(code, inp)
            status, diff = judge_output(result, expected)
            if status != "Accepted":
                if log_level == "DEBUG":
                    click.echo(
                        f"Test failed for input {inp}:\nExpected:\n{expected}\nGot:\n{result}\nDiff:\n{diff}\nStderr:\n{stderr}\n"
                    )
                submission_failed = True
            else:
                if log_level == "DEBUG":
                    click.echo(f"Test passed for input {inp}")
        if submission_failed:
            failed += 1
            if log_level == "DEBUG":
                click.echo(f"Submission {i} failed some tests.")
        else:
            passed += 1
            if log_level == "DEBUG":
                click.echo(f"Submission {i} passed all tests.")

    click.echo(f"\nSummary: Passed: {passed}, Failed: {failed}")
    return passed, failed


@click.command()
@click.option(
    "--response-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to JSONL file with fields: source, response, contest_id, index.",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
    help="Set the logging level.",
)
def main_evaluate(response_file, log_level):
    """
    Evaluate generated code responses against the experiment dataset tests.
    """
    evaluate_responses(response_file, log_level)


if __name__ == "__main__":
    main_evaluate()
