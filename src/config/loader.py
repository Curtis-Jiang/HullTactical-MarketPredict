"""Configuration loading utilities for HTMP pipelines."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(*paths: Path) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    for path in paths:
        if path is None:
            continue
        config = deep_merge(config, load_yaml(path))
    return config


def cfg_hash(payload: Dict[str, Any]) -> str:
    normalised = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(normalised).hexdigest()[:8]


__all__ = ["cfg_hash", "deep_merge", "load_config", "load_yaml"]
