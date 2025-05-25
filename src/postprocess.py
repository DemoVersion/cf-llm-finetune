def postprocess_response(response: str) -> str:
    """
    Postprocess the response from the model to ensure it is in the correct format.

    Args:
        response (str): The raw response from the model.

    Returns:
        str: The postprocessed response.
    """
    response = response.replace("```python", "```")
    text = response.replace("```python", "```")
    code_mark = "```"

    first_index = text.find(code_mark)
    last_index = text.rfind(code_mark)

    # Make sure both occurrences exist and are not the same
    if first_index != -1 and last_index != -1 and first_index != last_index:
        between = text[first_index + len(code_mark) : last_index]
        return between.strip()
    else:
        return text.strip()
