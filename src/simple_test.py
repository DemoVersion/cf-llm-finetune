import difflib

from tqdm import tqdm

from src.code_runner import run_script
from src.dataset import get_dataset
from src.generate import generate_code
from src.postprocess import postprocess_response


def judge_output(stdout: str, expected: str) -> str:
    stdout_lines = stdout.strip().splitlines()
    expected_lines = expected.strip().splitlines()

    if stdout_lines == expected_lines:
        return "Accepted", ""
    else:
        diff = difflib.unified_diff(expected_lines, stdout_lines, lineterm="")
        diffs = "\n".join(diff)
        return "Wrong Answer", diffs


def experiment():
    dataset = get_dataset()
    print(f"Dataset size: {len(dataset)}")
    cnt = 0
    passed = 0
    failed = 0
    for i, row in tqdm(dataset.iterrows()):
        # response = generate_code(row["source"], mode="local")
        response = generate_code(row["source"], mode="openai")
        extracted_code = postprocess_response(response)
        failed_test = False
        for test in row["examples"]:
            print(f"Running test: {test['input']}")
            result, stderr = run_script(extracted_code, test["input"])
            judge_result, diff = judge_output(result, test["output"])
            if judge_result != "Accepted":
                print(
                    f"Test failed for input {test['input']}: expected {test['output']}, got {result}, diff:\n{diff}, stderr: {stderr}"
                )
                failed_test = True
            else:
                print(f"Test passed for input {test['input']}")
        cnt += 1
        if not failed_test:
            passed += 1
            print(f"Submission {i} passed all tests.")
        else:
            failed += 1
            print(f"Submission {i} failed some tests.")
        if cnt >= 3:
            break
    print(f"Total submissions: {cnt}, Passed: {passed}, Failed: {failed}")


if __name__ == "__main__":
    experiment()
