import re

import pandas as pd
from datasets import load_dataset
from joblib import Memory

memory = Memory("./cache", verbose=0)


@memory.cache
def get_submissions_df(sample_size: int = 100000) -> pd.DataFrame:
    """
    Load a dataset of Codeforces submissions filtered for C++ programming language.
    Returns a DataFrame with a sample of 100,000 submissions.
    """
    ds = load_dataset("open-r1/codeforces-submissions", "default")
    results = ds["train"].filter(lambda x: "C++" in str(x["programmingLanguage"]))
    shuffled_results = results.shuffle(seed=42)
    selected_results = shuffled_results.select(range(sample_size))
    return selected_results.to_pandas()


@memory.cache
def get_dataset() -> pd.DataFrame:
    """
    Load a dataset of Codeforces problems and submissions, merging them based on contest ID and problem index.
    Returns a DataFrame with a sample of 2000 merged problems and submissions.
    """
    submission_df = get_submissions_df()
    submission_df_filtered = submission_df[
        ~submission_df["source"].str.contains(
            r"^\s*#\s*(define|ifdef)\b", flags=re.IGNORECASE, regex=True
        )
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
