import subprocess
from joblib import Memory

memory = Memory("./cache", verbose=0)


@memory.cache
def run_script(script, input_data=""):
    """
    Executes a Python script as a subprocess, providing input_data to its stdin,
    and captures its stdout and stderr outputs.

    Args:
        script (str): The Python script to execute.
        input_data (str): Input data to pass to the script's stdin.

    Returns:
        tuple: A tuple containing (stdout, stderr) outputs as strings.
    """
    try:
        result = subprocess.run(
            ["pipenv", "run", "python", "-c", script],
            input=input_data,
            text=True,
            capture_output=True,
            timeout=5,
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        return "", f"TimeoutExpired: {str(e)}"
