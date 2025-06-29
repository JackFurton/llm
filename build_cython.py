#!/usr/bin/env python3
"""
Build and test the Cython extensions

IMPORTANT: This script requires a virtual environment with Cython installed.
Run the following commands before executing this script:
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
"""
import os
import sys
import time

# Check if running in a virtual environment
if not hasattr(sys, 'base_prefix') or sys.base_prefix == sys.prefix:
    print("WARNING: This script should be run in a virtual environment.")
    print("Please create and activate a virtual environment first:")
    print("    python -m venv venv")
    print("    source venv/bin/activate")
    print("    pip install -r requirements.txt")
    print("Then run this script again.")
    sys.exit(1)

try:
    from Cython.Build import cythonize
    from setuptools import Extension
    import pyximport
except ImportError:
    print("ERROR: Cython is not installed. Please install it with:")
    print("    pip install cython")
    sys.exit(1)

# Initialize Cython
pyximport.install()

# Build the extension
try:
    extensions = [
        Extension(
            "src.cython_ext.fast_tokenizer",
            ["src/cython_ext/fast_tokenizer.pyx"],
        ),
    ]

    cythonize(extensions, language_level=3)
    print("Cython extension built successfully!")
except Exception as e:
    print(f"ERROR: Failed to build Cython extension: {e}")
    sys.exit(1)

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
