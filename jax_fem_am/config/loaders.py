"""Flat YAML/JSON config loading helpers.

Extracted verbatim from legacy/v03/am_thermal_stress_macro_intersection_mech100.py.
"""

import json


def parse_scalar(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return ""
    if text.lower() in ("true", "yes", "on"):
        return True
    if text.lower() in ("false", "no", "off"):
        return False
    if text.lower() in ("none", "null"):
        return None
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text.strip("\"'")


def read_config(path):
    if path is None:
        return {}
    with open(path) as f:
        text = f.read()
    if path.endswith(".json"):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("--config JSON must contain an object")
        return data

    data = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Only flat YAML key: value entries are supported, got: {line}")
        key, value = stripped.split(":", 1)
        data[key.strip().replace("-", "_")] = parse_scalar(value)
    return data


def cfg(config, key, default):
    return config.get(key, default)
