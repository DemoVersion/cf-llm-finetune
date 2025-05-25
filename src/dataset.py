import pandas as pd

from datasets import load_dataset


def get_dataset():
    submission_df = pd.read_csv("codeforces_cpp_submissions.csv")
    submission_df_filtered = submission_df[
        ~submission_df["source"].str.contains("#define")
    ].copy()
    ds = load_dataset("open-r1/codeforces", "default")
    problems_df = ds["train"].to_pandas()
    problems_df_filtered = problems_df[problems_df["generated_checker"].isna()].copy()

    problems_df_filtered["contest_id"] = problems_df_filtered["contest_id"].astype(str)
    problems_df_filtered["index"] = problems_df_filtered["index"].astype(str)
    submission_df_filtered["contestId"] = submission_df_filtered["contestId"].astype(
        str
    )
    submission_df_filtered["problem_index"] = submission_df_filtered[
        "problem_index"
    ].astype(str)
    merged_df = pd.merge(
        problems_df_filtered,
        submission_df_filtered,
        left_on=["contest_id", "index"],
        right_on=["contestId", "problem_index"],
        how="inner",
        suffixes=("_p", "_s"),
    )
    sampled_df = merged_df.sample(n=2000, random_state=42).copy()
    return sampled_df
