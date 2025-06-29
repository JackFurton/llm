# Custom Language Model from Scratch

A simple but complete implementation of a transformer-based language model built from scratch using PyTorch. This project demonstrates the core concepts of modern language models like GPT in a clean, educational codebase.

## Quick Start

```bash
# Setup the environment
./llm.sh setup

# Train a model (small and quick)
./llm.sh train --epochs 5

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
- **Training Pipeline**: Complete with checkpointing, evaluation, and visualization
- **Customizable**: Adjust model size, training parameters, and generation settings

## Usage

The project includes a single command-line tool `llm.sh` that handles all functionality:

### Setup

```bash
./llm.sh setup
```

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
│   ├── training/             # Training utilities
│   │   └── trainer.py        # Training loop implementation
│   ├── evaluation/           # Evaluation utilities
│   │   └── metrics.py        # Evaluation metrics
│   └── main.py               # Main script for training/generation
├── notebooks/                # Jupyter notebooks
├── tests/                    # Unit tests
├── data/                     # Data storage
│   ├── raw/                  # Raw text files
│   └── processed/            # Processed data
└── checkpoints/              # Model checkpoints (created during training)
```

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

1. **Use BPE tokenization**: `--tokenizer bpe --vocab-size 5000`
2. **Train a larger model**: `--d-model 256 --layers 6`
3. **Use beam search for coherent text**: `--beam --beam-size 5`
4. **Use high temperature for creative text**: `--temperature 1.2`
5. **Use low temperature for focused text**: `--temperature 0.5`

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Other dependencies in `requirements.txt`

## License

MIT
