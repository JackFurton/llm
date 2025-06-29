# Cython extensions for performance-critical components
import pyximport
pyximport.install()

try:
    from .fast_tokenizer import FastCharacterTokenizer
except ImportError:
    import os
    import sys
    print("Warning: Could not import Cython extension. Using Python implementation instead.")
    
    # Fall back to Python implementation
    from src.data.tokenizer import CharacterTokenizer as FastCharacterTokenizer
