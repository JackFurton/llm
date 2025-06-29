# Custom Language Model from Scratch

A simple but complete implementation of a transformer-based language model built from scratch using PyTorch. This project demonstrates the core concepts of modern language models like GPT in a clean, educational codebase.

## Getting Started

```bash
# Clone the repository
git clone https://github.com/JackFurton/llm.git
cd llm

# Setup the environment
./llm.sh setup

# Collect training data
./llm.sh collect --sources wikipedia --query "artificial intelligence" --limit 5

# Preprocess the collected data
./llm.sh preprocess --normalize --clean-html --clean-whitespace

# Curate the data using the web interface
./llm.sh curate

# Train a model
./llm.sh train --epochs 10

# Generate text
./llm.sh generate --prompt "Once upon a time"
```

## Project Overview

This project provides a complete pipeline for building and using a custom language model:

1. **Data Collection**: Gather text from Wikipedia, Project Gutenberg, news sources, and Reddit
2. **Data Preprocessing**: Clean, normalize, filter, and augment text data
3. **Data Curation**: Web interface for reviewing and selecting high-quality training data
4. **Model Training**: Train a transformer-based language model on the curated data
5. **Text Generation**: Generate text using the trained model

All functionality is accessible through a single command-line tool (`llm.sh`).

## Usage

### Data Collection

```bash
# Collect from Wikipedia
./llm.sh collect --sources wikipedia --query "machine learning" --limit 10

# Collect from Project Gutenberg
./llm.sh collect --sources gutenberg --limit 5

# Collect from multiple sources
./llm.sh collect --sources wikipedia gutenberg news --query "artificial intelligence" --limit 3
```

### Data Preprocessing

```bash
# Basic preprocessing
./llm.sh preprocess --normalize --clean-html --clean-whitespace

# Language filtering
./llm.sh preprocess --language en

# Text augmentation
./llm.sh preprocess --normalize --augment --synonym-replace
```

### Data Curation

```bash
# Start the web interface
./llm.sh curate

# Start on a specific host and port
./llm.sh curate --host 0.0.0.0 --port 8080
```

### Training

```bash
# Train a small model quickly
./llm.sh train --epochs 5

# Train with BPE tokenizer for better results
./llm.sh train --tokenizer bpe --vocab-size 5000 --d-model 128 --layers 4 --epochs 20
```

### Generation

```bash
# Generate text with default settings
./llm.sh generate --prompt "Hello world"

# Generate with beam search for more coherent text
./llm.sh generate --prompt "The future of AI" --beam --beam-size 5

# Generate with sampling for more creative text
./llm.sh generate --prompt "The future of AI" --temperature 1.2
```

## Performance Optimizations

This project includes Cython optimizations for performance-critical components like tokenization. These optimizations provide significant speedups (up to 1.8x faster for encoding and 1.4x for decoding).

**Note:** The Cython extensions require a virtual environment with Cython installed. To run the benchmark:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the benchmark
python benchmark_tokenizers.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Flask 2.3+ (for web interface)
- Cython 3.0+ (for performance optimizations)
- Other dependencies in `requirements.txt`

## License

MIT
