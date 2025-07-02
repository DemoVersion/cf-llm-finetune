import difflib
from typing import Union


def judge_output(stdout: str, expected: str) -> Union[str, str]:
    """
    Compare actual stdout to expected, return status and diff if any.
    """
    stdout_lines = stdout.strip().splitlines()
    expected_lines = expected.strip().splitlines()

    if stdout_lines == expected_lines:
        return "Accepted", ""
    diff = difflib.unified_diff(expected_lines, stdout_lines, lineterm="")
    return "Wrong Answer", "\n".join(diff)
