"""Backwards compatible config helpers (deprecated)."""

from __future__ import annotations

from src.config.loader import cfg_hash, deep_merge, load_config, load_yaml  # noqa: F401

__all__ = ["cfg_hash", "deep_merge", "load_config", "load_yaml"]
