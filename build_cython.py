#!/usr/bin/env python3
"""
Build and test the Cython extensions
"""
import os
import sys
import time
from Cython.Build import cythonize
from setuptools import Extension
import pyximport

# Initialize Cython
pyximport.install()

# Build the extension
extensions = [
    Extension(
        "src.cython_ext.fast_tokenizer",
        ["src/cython_ext/fast_tokenizer.pyx"],
    ),
]

cythonize(extensions, language_level=3)

print("Cython extension built successfully!")

# Test the extension
try:
    from src.cython_ext.fast_tokenizer import FastCharacterTokenizer
    from src.data.tokenizer import CharacterTokenizer
    
    # Create test data
    test_texts = [
        "Hello, world!",
        "Testing character tokenization.",
        "1234567890!@#$%^&*()"
    ]
    
    # Test Cython tokenizer
    print("\nTesting Cython tokenizer...")
    start_time = time.time()
    
    cython_tokenizer = FastCharacterTokenizer()
    cython_tokenizer.train(test_texts)
    
    for text in test_texts:
        encoded = cython_tokenizer.encode(text)
        decoded = cython_tokenizer.decode(encoded)
        print(f"Original: '{text}'")
        print(f"Encoded: {encoded}")
        print(f"Decoded: '{decoded}'")
        assert text == decoded, f"Decoding failed: '{text}' != '{decoded}'"
    
    cython_time = time.time() - start_time
    print(f"Cython tokenizer time: {cython_time:.6f} seconds")
    
    # Test Python tokenizer
    print("\nTesting Python tokenizer...")
    start_time = time.time()
    
    python_tokenizer = CharacterTokenizer()
    python_tokenizer.train(test_texts)
    
    for text in test_texts:
        encoded = python_tokenizer.encode(text)
        decoded = python_tokenizer.decode(encoded)
        print(f"Original: '{text}'")
        print(f"Encoded: {encoded}")
        print(f"Decoded: '{decoded}'")
        assert text == decoded, f"Decoding failed: '{text}' != '{decoded}'"
    
    python_time = time.time() - start_time
    print(f"Python tokenizer time: {python_time:.6f} seconds")
    
    # Compare performance
    speedup = python_time / cython_time
    print(f"\nCython speedup: {speedup:.2f}x")
    
except ImportError as e:
    print(f"Error importing Cython extension: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error testing Cython extension: {e}")
    sys.exit(1)

print("\nCython extension test completed successfully!")
