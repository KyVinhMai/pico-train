"""
Model Package

This Package contains Pico models (currently only the Pico Decoder). We plan to implement other
architectures in the future.

If you have other models you'd like to implement, we recommend you add modules to this package.
"""

from .sentencepiece_wrapper import SentencePieceTokenizerWrapper

__all__ = ["SentencePieceTokenizerWrapper"]
