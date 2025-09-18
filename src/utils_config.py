import yaml, json, hashlib
from copy import deepcopy
from pathlib import Path

def load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(*paths: Path) -> dict:
    cfg = {}
    for p in paths:
        if p is not None: cfg = deep_merge(cfg, load_yaml(p))
    return cfg

def cfg_hash(d: dict) -> str:
    payload = json.dumps(d, sort_keys=True, separators=(",",":")).encode()
    return hashlib.sha256(payload).hexdigest()[:8]