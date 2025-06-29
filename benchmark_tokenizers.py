#!/usr/bin/env python3
"""
Benchmark the performance of Python vs Cython tokenizers
"""
import os
import sys
import time
import random
import string
import pyximport

# Initialize Cython
pyximport.install()

# Import tokenizers
from src.data.tokenizer import CharacterTokenizer
from src.cython_ext.fast_tokenizer import FastCharacterTokenizer

def generate_random_text(length):
    """Generate random text of specified length"""
    return ''.join(random.choice(string.ascii_letters + string.digits + string.punctuation + ' ') for _ in range(length))

def benchmark_tokenizer(tokenizer_class, texts, num_iterations=100):
    """Benchmark a tokenizer on the given texts"""
    # Train the tokenizer
    tokenizer = tokenizer_class()
    tokenizer.train(texts)
    
    # Benchmark encoding
    start_time = time.time()
    for _ in range(num_iterations):
        for text in texts:
            encoded = tokenizer.encode(text)
    encode_time = time.time() - start_time
    
    # Benchmark decoding
    start_time = time.time()
    for _ in range(num_iterations):
        for text in texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
    decode_time = time.time() - start_time - encode_time
    
    return {
        "encode_time": encode_time,
        "decode_time": decode_time,
        "total_time": encode_time + decode_time
    }

def main():
    # Generate test data
    print("Generating test data...")
    num_texts = 1000
    text_length = 1000
    texts = [generate_random_text(text_length) for _ in range(num_texts)]
    
    print(f"Generated {num_texts} texts of length {text_length}")
    
    # Benchmark Python tokenizer
    print("\nBenchmarking Python tokenizer...")
    python_results = benchmark_tokenizer(CharacterTokenizer, texts)
    
    # Benchmark Cython tokenizer
    print("Benchmarking Cython tokenizer...")
    cython_results = benchmark_tokenizer(FastCharacterTokenizer, texts)
    
    # Print results
    print("\nResults:")
    print(f"Python tokenizer:")
    print(f"  Encode time: {python_results['encode_time']:.6f} seconds")
    print(f"  Decode time: {python_results['decode_time']:.6f} seconds")
    print(f"  Total time: {python_results['total_time']:.6f} seconds")
    
    print(f"\nCython tokenizer:")
    print(f"  Encode time: {cython_results['encode_time']:.6f} seconds")
    print(f"  Decode time: {cython_results['decode_time']:.6f} seconds")
    print(f"  Total time: {cython_results['total_time']:.6f} seconds")
    
    # Calculate speedup
    encode_speedup = python_results['encode_time'] / cython_results['encode_time']
    decode_speedup = python_results['decode_time'] / cython_results['decode_time']
    total_speedup = python_results['total_time'] / cython_results['total_time']
    
    print(f"\nSpeedup:")
    print(f"  Encode: {encode_speedup:.2f}x")
    print(f"  Decode: {decode_speedup:.2f}x")
    print(f"  Total: {total_speedup:.2f}x")

if __name__ == "__main__":
    main()
