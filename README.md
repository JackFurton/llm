# Custom Language Model from Scratch

A simple but complete implementation of a transformer-based language model built from scratch using PyTorch. This project demonstrates the core concepts of modern language models like GPT in a clean, educational codebase.

## Quick Start

```bash
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

# Evaluate model performance
./llm.sh evaluate
```

## Features

- **Transformer Architecture**: Multi-head attention, positional encoding, and feed-forward networks
- **Multiple Tokenizers**: Character-level, word-level, and BPE tokenization options
- **Advanced Text Generation**: Temperature sampling, top-k, top-p, and beam search
- **Model Evaluation**: Perplexity, BLEU score, and diversity metrics
- **Data Collection**: Gather training data from Wikipedia, Project Gutenberg, news sources, and Reddit
- **Data Preprocessing**: Clean, normalize, filter, and augment text data
- **Data Curation**: Web interface for reviewing and selecting high-quality training data
- **Training Pipeline**: Complete with checkpointing, evaluation, and visualization
- **Customizable**: Adjust model size, training parameters, and generation settings

## Usage

The project includes a single command-line tool `llm.sh` that handles all functionality:

### Setup

```bash
./llm.sh setup
```

### Data Collection

Collect training data from various sources:

```bash
# Collect from Wikipedia
./llm.sh collect --sources wikipedia --query "machine learning" --limit 10

# Collect from Project Gutenberg
./llm.sh collect --sources gutenberg --limit 5

# Collect from Reddit
./llm.sh collect --sources reddit --subreddit "science" --limit 5

# Collect from multiple sources
./llm.sh collect --sources wikipedia gutenberg news --query "artificial intelligence" --limit 3
```

### Data Preprocessing

Preprocess the collected data to improve quality:

```bash
# Basic preprocessing
./llm.sh preprocess --normalize --clean-html --clean-whitespace

# Language filtering
./llm.sh preprocess --language en

# Content filtering
./llm.sh preprocess --normalize --no-content-filter

# Text augmentation
./llm.sh preprocess --normalize --augment --synonym-replace
```

### Data Curation

Launch the web interface for data curation:

```bash
# Start the web interface
./llm.sh curate

# Start on a specific host and port
./llm.sh curate --host 0.0.0.0 --port 8080

# Start in debug mode
./llm.sh curate --debug
```

Then open your browser and navigate to http://127.0.0.1:5000 (or the host/port you specified).

### Training

Train a small model quickly:
```bash
./llm.sh train --epochs 5
```

Train with BPE tokenizer for better results:
```bash
./llm.sh train --tokenizer bpe --vocab-size 5000 --d-model 128 --layers 4 --epochs 20
```

### Generation

Generate text with default settings:
```bash
./llm.sh generate --prompt "Hello world"
```

Generate with beam search for more coherent text:
```bash
./llm.sh generate --prompt "The future of AI" --beam --beam-size 5
```

Generate with sampling for more creative text:
```bash
./llm.sh generate --prompt "The future of AI" --temperature 1.2
```

### Evaluation

Evaluate model performance:
```bash
./llm.sh evaluate
```

Evaluate on specific data:
```bash
./llm.sh evaluate --eval-data path/to/data.txt --eval-samples 10
```

### Testing

Run the test suite:
```bash
./llm.sh test
```

## Project Structure

```
llm-project/
├── llm.sh                    # Main command-line tool
├── src/                      # Source code
│   ├── model/                # Model architecture
│   │   ├── transformer.py    # Transformer implementation
│   │   └── generation.py     # Advanced text generation methods
│   ├── data/                 # Data processing
│   │   ├── tokenizer.py      # Basic tokenization utilities
│   │   ├── bpe_tokenizer.py  # BPE tokenization
│   │   └── dataset.py        # Dataset classes
│   ├── data_collection/      # Data collection utilities
│   │   ├── collector.py      # Base collector framework
│   │   └── sources.py        # Data source implementations
│   ├── data_preprocessing/   # Data preprocessing utilities
│   │   ├── preprocessor.py   # Base preprocessor framework
│   │   ├── filters.py        # Text filters
│   │   ├── normalizers.py    # Text normalizers
│   │   └── augmenters.py     # Text augmenters
│   ├── web_interface/        # Web interface for data curation
│   │   ├── app.py            # Flask application
│   │   ├── run.py            # Script to run the web interface
│   │   ├── templates/        # HTML templates
│   │   └── static/           # Static files (CSS, JS)
│   ├── training/             # Training utilities
│   │   └── trainer.py        # Training loop implementation
│   ├── evaluation/           # Evaluation utilities
│   │   └── metrics.py        # Evaluation metrics
│   └── main.py               # Main script for training/generation
├── notebooks/                # Jupyter notebooks
├── tests/                    # Unit tests
├── data/                     # Data storage
│   ├── raw/                  # Raw text files
│   ├── processed/            # Processed data
│   └── curated/              # Curated data for training
└── checkpoints/              # Model checkpoints (created during training)
```

## Data Collection Sources

The project includes several data sources for collecting training data:

- **Wikipedia**: Collects articles from Wikipedia based on search queries
- **Project Gutenberg**: Collects public domain books from Project Gutenberg
- **News**: Collects news articles from various RSS feeds
- **Reddit**: Collects posts and comments from specified subreddits

## Data Preprocessing Features

The project includes several preprocessing capabilities:

### Filters

- **Language Filter**: Filter text based on detected language
- **Content Filter**: Filter text containing profanity or sensitive topics
- **Quality Filter**: Filter text based on quality metrics (length, word length, etc.)
- **Duplicate Filter**: Filter duplicate or near-duplicate text

### Normalizers

- **Text Normalizer**: Apply basic text normalization (punctuation, whitespace, etc.)
- **HTML Cleaner**: Clean HTML tags and entities
- **Whitespace Cleaner**: Clean and normalize whitespace
- **Markdown Cleaner**: Clean Markdown formatting

### Augmenters

- **Synonym Replacer**: Replace words with synonyms
- **Back Translator**: Translate text to another language and back

## Data Curation Interface

The web interface provides the following features:

- **File Browsing**: Browse raw, processed, and curated data files
- **File Viewing**: View the content of data files
- **File Editing**: Edit the content of data files
- **Curation**: Select high-quality processed files for training
- **Statistics**: View statistics about your data collection and preprocessing

## Model Architecture

The model is a decoder-only transformer with the following components:

- Token embeddings
- Positional encodings
- Multi-head self-attention
- Feed-forward networks
- Layer normalization

## Advanced Features

### Tokenization Options

- **Character-level**: Simple but requires more parameters
- **Word-level**: Better for small datasets
- **BPE (Byte-Pair Encoding)**: Best balance of vocabulary size and token meaning

### Text Generation Methods

- **Temperature Sampling**: Control randomness with temperature parameter
- **Top-K Sampling**: Only sample from the K most likely tokens
- **Top-P (Nucleus) Sampling**: Sample from the smallest set of tokens whose cumulative probability exceeds P
- **Beam Search**: Generate multiple candidate sequences and select the best one

### Evaluation Metrics

- **Perplexity**: Measures how well the model predicts the test data
- **BLEU Score**: Compares generated text to reference text
- **Diversity Metrics**: Measures the variety in generated text

## Tips for Better Results

1. **Collect diverse training data**: Use multiple sources with `./llm.sh collect`
2. **Preprocess your data**: Clean and normalize with `./llm.sh preprocess`
3. **Curate your data**: Select high-quality examples with `./llm.sh curate`
4. **Use BPE tokenization**: `--tokenizer bpe --vocab-size 5000`
5. **Train a larger model**: `--d-model 256 --layers 6`
6. **Use beam search for coherent text**: `--beam --beam-size 5`
7. **Use high temperature for creative text**: `--temperature 1.2`
8. **Use low temperature for focused text**: `--temperature 0.5`

## Performance Optimizations

The project includes several performance optimizations:

- **Cython Extensions**: Performance-critical components like tokenization are implemented in Cython for improved speed (up to 1.7x faster)
- **Efficient Data Processing**: Streaming data loading to handle large datasets
- **Batched Operations**: Processing data in batches for better GPU utilization
- **Memory Efficiency**: Careful management of memory usage during training and inference

### Using the Cython Tokenizer

To use the Cython-optimized tokenizer in your code:

```python
from src.cython_ext import FastCharacterTokenizer

# Create and train the tokenizer
tokenizer = FastCharacterTokenizer()
tokenizer.train(["Sample text for training"])

# Encode and decode text
encoded = tokenizer.encode("Hello, world!")
decoded = tokenizer.decode(encoded)
```

### Running Benchmarks

To run the Cython performance benchmark:

```bash
python benchmark_tokenizers.py
```

To test the Cython extension:

```bash
python build_cython.py
```

For a simple example of using the Cython tokenizer:

```bash
python use_cython_tokenizer.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Flask 2.3+
- Cython 3.0+ (for performance optimizations)
- Other dependencies in `requirements.txt`

## License

MIT
