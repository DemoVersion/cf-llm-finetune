apply_autoflake:
    uv run autoflake --in-place --remove-all-unused-imports --recursive .

apply_black:
    uv run black .

apply_isort:
    uv run isort --profile black .

clean_code: apply_autoflake apply_isort apply_black