#!/usr/bin/env python3
"""
Example of how to use the Cython-optimized tokenizer
"""
import time

# Import the Cython tokenizer
from src.cython_ext import FastCharacterTokenizer

def main():
    # Create a tokenizer
    tokenizer = FastCharacterTokenizer()
    
    # Train on some sample texts
    sample_texts = [
        "Hello, world!",
        "This is a test of the Cython-optimized tokenizer.",
        "It should be significantly faster than the Python version."
    ]
    
    tokenizer.train(sample_texts)
    
    # Encode and decode a text
    text = "Hello, Cython world!"
    
    start_time = time.time()
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    end_time = time.time()
    
    print(f"Original text: '{text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    print(f"Processing time: {(end_time - start_time) * 1000:.3f} ms")

if __name__ == "__main__":
    main()
