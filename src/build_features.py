#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI for building level-1 feature datasets."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.config import deep_merge
from src.features.pipeline import (
    DEFAULT_CFG,
    build_level1,
    build_version_manifest,
    ensure_dir,
    read_yaml_if_any,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build level-1 features for HTMP")
    parser.add_argument("--train", type=str, default="data/raw/train.csv")
    parser.add_argument("--test", type=str, default="data/raw/test.csv")
    parser.add_argument("--outdir", type=str, default="data/processed")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def load_feature_config(path: Path | None) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CFG)
    if path is not None:
        override = read_yaml_if_any(path)
        cfg = deep_merge(cfg, override)
    return cfg


def main() -> None:
    args = parse_args()

    train_path = Path(args.train).expanduser().resolve()
    test_path = Path(args.test).expanduser().resolve()
    out_root = Path(args.outdir).expanduser().resolve()
    cfg_path = Path(args.config).expanduser().resolve() if args.config else None

    cfg = load_feature_config(cfg_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X, T, meta = build_level1(train_df, test_df, cfg)
    version, manifest = build_version_manifest(train_path, test_path, cfg, meta)

    out_dir = out_root / f"htmp_v{version}"
    ensure_dir(out_dir)

    train_pq = out_dir / "train_l1.parquet"
    test_pq = out_dir / "test_l1.parquet"
    manifest_path = out_dir / "manifest.json"

    X.to_parquet(train_pq, index=False)
    T.to_parquet(test_pq, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[ok] saved Level-1 features to: {out_dir}")
    print(f"  train_l1.parquet: {X.shape}  test_l1.parquet: {T.shape}")
    print(
        "  safe_cols: {safe}  windows: {windows}  stats: {stats}".format(
            safe=len(meta.get("safe_cols", [])),
            windows=meta.get("rolling", {}).get("windows", []),
            stats=meta.get("rolling", {}).get("stats", []),
        )
    )


if __name__ == "__main__":
    main()
