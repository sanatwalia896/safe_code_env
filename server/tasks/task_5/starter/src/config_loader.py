import json


DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 8080,
    "debug": False,
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise TypeError("config must be a JSON object")

    merged = data.copy()
    merged.update(DEFAULT_CONFIG)
    return merged
