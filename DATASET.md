# cf-llm-finetune

## Dataset
`cf-llm-finetune` uses these two datasets to generate a synthetic parallel dataset for fine-tuning:
- `open-r1/codeforces-submissions`
    - C++ ICPC solutions from Codeforces submissions
- `open-r1/codeforces`
    - Problem statements and sample inputs

## Dataset Generation
We first get all the C++ ICPC solutions from the `open-r1/codeforces-submissions` dataset. We do that by filtering by "programmingLanguage" field. Then we sample 100 unique solutions per problem in this dataset to make the dataset smaller and more manageable. To do this efficiently, we create chunks of 500,000 solutions and then sample 100 unique solutions per problem from each chunk. Then at the end, we merge all the chunks together and sample 100 unique solutions per problem again.
In the following cleaning step, we remove remove the solutions containing macros like `#define`, `#ifdef`, and `#ifndef` to ensure the that we don't get any macro-heavy solutions. Because the macro-heavy solutions could lead to almost like a new programming language other than C++, we want to avoid them. We believe translating such solutions would require a preprocessor to handle the macros, which is not the goal of this project.
Then we merge this dataset with the `open-r1/codeforces` dataset to get the problem statements and sample inputs, and keep only the one unique solution per problem. This ensures that we have only one solution per problem, and not a single problem with multiple solutions is leaked between the training and test datasets.
This also balances the dataset, because most of the solutions are for the easier problems and we want to see how well the model can perform in all ranges of problems which is achievable by unique solutions per problem.
Finally, we sample 2000 unique problems from this dataset to create the final dataset. This is done to ensure that we have a manageable dataset size for training and testing. Then we seperate the solutions into train, validation, and test datasets. The train dataset contains 1400 unique problems, the validation dataset contains 300 unique problems, and the test dataset contains 300 unique problems. The code for the dataset generation could be found in the `src/dataset.py` file.

## Evaluation
Evaluation is done by running the generated Python solutions against the sample inputs provided in the problem statements. Although it is not a good way to test and any bruteforce solution will pass the tests, it is a the common way to evaluate the solutions in competitive programming as noted here by [DeepSeek](https://huggingface.co/datasets/open-r1/codeforces) team as well.
The evaluation is done by compiling and running the generated Python solutions against the sample inputs provided in the problem statements. The evaluation is done using a test harness that compiles and runs the Python output against provided samples. The code for the evaluation could be found in the `src/evaluate_response.py` file. 
Here we use a simple test harness that compiles and runs the Python output against provided samples. If you don't trust the generated Python solutions, you should modify the `src/evaluate_response.py` file to use a sandbox like [judge0](https://github.com/judge0/judge0) to run the generated Python solutions. The current implementation is not secure and should not be used in production environments.

## Solution Generation for Synthetic Dataset
We use OpenAI's GPT-4.1 to generate Python translations for the C++ solutions in the dataset. The goal is to create a synthetic parallel dataset that can be used for fine-tuning the model. The generated Python translations are then paired with the original C++ code to form a fine-tuning corpus. The code for the solution generation could be found in the `src/generate_openai.py` file.
This file provides both direct OpenAI API calls and a batch job submission mode for generating the solutions. The batch job submission mode is useful for generating large datasets for saving costs and avoiding rate limits. The generated solutions are saved in a JSONL file format, which is then used for fine-tuning the model.
Before running the solution generation, make sure to set the `OPENAI_API_KEY` environment variable with your OpenAI API key. You can do this by creating a `.env` file in the root directory of the project and adding the following line:
```
OPENAI_KEY=your_openai_api_key_here
```
Then you can run the solution generation script using the following command for train, dataset:
```bash
uv run --env-file .env python -m src.generate_openai --dataset-name train
```
You need to repeat this for the validation and test datasets by changing the `--dataset-name` argument to `val` and `test` respectively. You can also use `functional-test` it consists of 10 unique problems from validation dataset to just be sure that the script is working correctly without generating a large dataset.
Note that first time you run the script, it will take a while to generate dataset because it will create the dataset from scratch. But then it will get cached and subsequent runs will be much faster.
