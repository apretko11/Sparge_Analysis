"""Prefix-safe, ALiBi-aware sparse attention for the audited Celerity model."""

from .adapter import CeleritySparseAdapter
from .attention import SparseSettings

__all__ = ["CeleritySparseAdapter", "SparseSettings"]
