"""Config loading: YAML -> nested SimpleNamespace with dotted get/set helpers."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_ns(v) for v in obj]
    return obj


def ns_to_dict(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: ns_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [ns_to_dict(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = to_ns(raw)
    cfg._source = str(path)
    return cfg


def get(cfg: Any, dotted: str, default: Any = None) -> Any:
    """cfg, 'model.name' -> value (or default)."""
    obj = cfg
    for part in dotted.split("."):
        if obj is None:
            return default
        if isinstance(obj, SimpleNamespace):
            obj = getattr(obj, part, None)
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return default
    return default if obj is None else obj


def set_(cfg: Any, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    obj = cfg
    for part in parts[:-1]:
        if isinstance(obj, SimpleNamespace):
            obj = getattr(obj, part)
        else:
            obj = obj[part]
    last = parts[-1]
    if isinstance(obj, SimpleNamespace):
        setattr(obj, last, value)
    else:
        obj[last] = value


def save_config(cfg: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ns_to_dict(cfg), f, indent=2, sort_keys=True)
