import os
import yaml
import json

__all__ = [
    "load",
    "load_json",
    "load_jsonl",
    "load_txt",
    "load_yaml",
    "load_raw",
    "get_filename_without_ext",
]


def load(path: str) -> dict|str|list:
    if path.endswith('.json'):
        return load_json(path)
    elif path.endswith('.jsonl'):
        return load_jsonl(path)
    elif path.endswith('.txt'):
        return load_txt(path)
    elif path.endswith('.yaml') or path.endswith('.yml'):
        return load_yaml(path)
    else:
        return load_raw(path)


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def load_jsonl(path: str) -> list:
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_raw(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()


def load_txt(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_filename_without_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
