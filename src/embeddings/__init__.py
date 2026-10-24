"""CLAP embedding generation."""

from .clap import ClapEncoder
from .index import load_index, save_index

__all__ = ["ClapEncoder", "load_index", "save_index"]
