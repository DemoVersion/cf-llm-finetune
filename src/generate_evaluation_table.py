import click
import pandas as pd

from src.evaluate_responses import evaluate_responses


@click.command()
@click.argument("response_files", nargs=-1)
def generate_evaluation_table(response_files):
    """
    Generate an evaluation table from the response files.
    """
    results = []
    for response_file in response_files:
        click.echo(f"Processing response file: {response_file}")
        passed, failed = evaluate_responses(response_file=response_file)
        success_rate = passed / (passed + failed) if (passed + failed) > 0 else 0.0
        results.append(
            {
                "response_file": response_file,
                "passed": passed,
                "failed": failed,
                "total": passed + failed,
                "success_rate": success_rate,
                "success_percentage": "{:.2%}".format(success_rate),
            }
        )
    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results.csv", index=False)
    click.echo("Evaluation results saved to evaluation_results.csv")


if __name__ == "__main__":
    generate_evaluation_table()
