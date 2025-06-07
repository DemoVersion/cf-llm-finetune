import os
import re

import pandas as pd
from datasets import load_dataset
from joblib import Memory
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.logger import logger

memory = Memory("./cache", verbose=0)


def load_experiment_dataset():
    """
    Load the dataset from a pickle file if it exists, otherwise load and merge problems and submissions.
    Returns a DataFrame containing the dataset.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    jsonl_file = os.path.join(dir_path, "dataset.jsonl")
    if os.path.exists(jsonl_file):
        print("Loading dataset from jsonl file...")
        dataset = pd.read_json(jsonl_file, orient="records", lines=True)
        return dataset
    dataset = load_and_merge_problems_submissions()
    dataset.to_json("dataset.jsonl", orient="records", lines=True)
    return dataset


def load_dataset_split():
    """
    Load the dataset and split it into train, validation, and test sets.
    Returns a tuple of DataFrames: (train_df, val_df, test_df).
    """
    dataset = load_experiment_dataset()
    train_df, val_df, test_df = split_dataset(dataset)
    return train_df, val_df, test_df


def get_limited_submissions_per_problem(
    cpp_only_submissions, max_submissions=100
) -> pd.DataFrame:
    """
    Get a maximum of 100 submissions per unique problem from the DataFrame.
    """
    chunk_size = 500_000  # Tune this based on your RAM
    num_chunks = len(cpp_only_submissions) // chunk_size + 1

    filtered_chunks = []
    for i in tqdm(range(num_chunks)):
        # Get slice
        chunk = cpp_only_submissions.select(
            range(i * chunk_size, min((i + 1) * chunk_size, len(cpp_only_submissions)))
        )

        # Convert to Pandas
        df = chunk.to_pandas()
        # Create the key
        df["key"] = df["contestId"].astype(str) + "_" + df["problem_index"].astype(str)

        # Apply groupby().head(100)
        df_filtered = df.groupby("key").head(10).reset_index(drop=True)
        filtered_chunks.append(df_filtered)
        logger.info(
            f"Processed chunk {i + 1}/{num_chunks}, size after filtering: {df_filtered.shape[0]}"
        )
    # Concatenate all filtered chunks
    filtered_df = pd.concat(filtered_chunks, ignore_index=True)
    logger.info(f"Total size after initial filtering: {filtered_df.shape[0]}")
    # filter again to ensure we have at most 100 submissions per unique problem
    filtered_df = (
        filtered_df.groupby("key").head(max_submissions).reset_index(drop=True)
    )
    logger.info(f"Final size after limiting submissions: {filtered_df.shape[0]}")
    return filtered_df.drop(columns=["key"]).copy()


@memory.cache
def load_cpp_submissions(max_submissions=100) -> pd.DataFrame:
    """
    Load a dataset of Codeforces submissions filtered for C++ programming language.
    Returns a DataFrame with limited number of submissions per unique problem.
    """
    raw_submission_dataset = load_dataset("open-r1/codeforces-submissions", "default")
    cpp_only_submissions = raw_submission_dataset["train"].filter(
        lambda x: "C++" in str(x["programmingLanguage"])
    )
    cpp_only_submissions = cpp_only_submissions.shuffle(seed=42)

    # We want to sample up to 100 submissions per unique problem,
    # this way the easy problems will not dominate the dataset.
    sampled_cpp_submissions = get_limited_submissions_per_problem(
        cpp_only_submissions, max_submissions=max_submissions
    )

    return sampled_cpp_submissions


@memory.cache
def load_and_merge_problems_submissions() -> pd.DataFrame:
    """
    Load a dataset of Codeforces problems and submissions, merging them based on contest ID and problem index.
    Returns a DataFrame with a sample of 2000 merged problems and submissions.
    """
    raw_submissions_df = load_cpp_submissions()
    logger.info(f"Raw submissions dataset size: {raw_submissions_df.shape[0]}")
    # Filter out macro-heavy source code
    filtered_submissions = raw_submissions_df[
        ~raw_submissions_df["source"].str.contains(
            r"^\s*#\s*(?:define|ifdef)\b", flags=re.IGNORECASE, regex=True
        )
    ].copy()

    raw_problems_dataset = load_dataset("open-r1/codeforces", "default")

    # Concatenate train and test datasets because we are splitting based on merged data later
    problems = pd.concat(
        [
            raw_problems_dataset["train"].to_pandas(),
            raw_problems_dataset["test"].to_pandas(),
        ],
        ignore_index=True,
    )
    logger.info(f"Raw problems dataset size: {problems.shape[0]}")
    filtered_problems = problems[problems["generated_checker"].isna()].copy()
    logger.info(f"Filtered problems dataset size: {filtered_problems.shape[0]}")
    merged_problems_submissions = pd.merge(
        filtered_problems,
        filtered_submissions,
        left_on=["contest_id", "index"],
        right_on=["contestId", "problem_index"],
        how="inner",
        suffixes=("_p", "_s"),
    )
    logger.info(
        f"Merged problems and submissions dataset size: {merged_problems_submissions.shape[0]}"
    )
    deduplicated_merged = merged_problems_submissions.drop_duplicates(
        subset=["contest_id", "index"], keep="first"
    )
    logger.info(f"Deduplicated merged dataset size: {deduplicated_merged.shape[0]}")
    sampled_df = deduplicated_merged.sample(n=2000, random_state=42).copy()
    return sampled_df


def split_dataset(
    df: pd.DataFrame, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
):
    """
    Split the dataset into train, validation, and test sets.
    """
    train_df, temp_df = train_test_split(
        df, train_size=train_size, random_state=random_state
    )
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_ratio, random_state=random_state
    )
    return train_df, val_df, test_df
