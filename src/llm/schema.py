"""YAML schema loading shared by the CLI and async client."""

import json
from pathlib import Path
import sys

import yaml


def check_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("schema mapping keys must be strings; quote YAML boolean or numeric keys")
            check_keys(item)
    elif isinstance(value, list):
        for item in value:
            check_keys(item)


def response_format(path: str) -> str:
    with Path(path).open(encoding="utf-8") as source:
        schema = yaml.safe_load(source)
    if not isinstance(schema, dict):
        raise ValueError("schema must be a YAML mapping containing JSON Schema")
    result = json.dumps(
        {"type": "json_schema", "json_schema": {"name": "response", "schema": schema}},
        allow_nan=False,
    )
    check_keys(schema)
    return result


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: schema.py SCHEMA_FILE", file=sys.stderr)
        return 2
    try:
        result = response_format(sys.argv[1])
    except (OSError, UnicodeError, yaml.YAMLError, TypeError, ValueError, RecursionError) as error:
        print(f"Invalid schema file {sys.argv[1]}: {error}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
