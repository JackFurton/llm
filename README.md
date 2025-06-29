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
```

## Features

- **Transformer Architecture**: Multi-head attention, positional encoding, and feed-forward networks
- **Tokenization**: Character-level and word-level tokenization options
- **Training Pipeline**: Complete with checkpointing, evaluation, and visualization
- **Text Generation**: Temperature-controlled sampling for creative text generation
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

Train a larger model for better results:
```bash
./llm.sh train --d-model 128 --layers 4 --epochs 20
```

### Generation

Generate text with default settings:
```bash
./llm.sh generate --prompt "Hello world"
```

Adjust generation parameters:
```bash
./llm.sh generate --prompt "The future of AI" --temperature 0.5 --max-length 200
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
│   │   └── transformer.py    # Transformer implementation
│   ├── data/                 # Data processing
│   │   ├── tokenizer.py      # Tokenization utilities
│   │   └── dataset.py        # Dataset classes
│   ├── training/             # Training utilities
│   │   └── trainer.py        # Training loop implementation
│   └── main.py               # Main script for training/generation
├── notebooks/                # Jupyter notebooks
│   └── model_exploration.ipynb  # Example notebook
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

## Customization

You can customize various aspects of the model:

- **Model size**: Adjust `--d-model`, `--layers`, and `--heads`
- **Training**: Change `--epochs`, `--batch-size`
- **Tokenization**: Choose between `--tokenizer char` or `--tokenizer word`
- **Generation**: Control creativity with `--temperature` (lower = more focused, higher = more random)

## Tips for Better Results

1. **Add more data**: Place text files in `data/raw/` directory
2. **Train longer**: Increase epochs (e.g., `--epochs 50`)
3. **Use a larger model**: Increase model size with `--d-model 256 --layers 6`
4. **Adjust temperature**: Lower values (0.1-0.5) for focused text, higher values (0.7-1.0) for creative text

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Other dependencies in `requirements.txt`

## License

MIT
