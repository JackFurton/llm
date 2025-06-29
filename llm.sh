#!/bin/bash

# Main script for the Custom LLM project
# Handles setup, training, and generation

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Setting up..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Default values
MODE="help"
DATA_DIR="data/raw"
TOKENIZER_TYPE="char"
EPOCHS=10
D_MODEL=64
NUM_LAYERS=2
NUM_HEADS=4
BATCH_SIZE=4
MODEL_PATH="checkpoints/best_model.pt"
TOKENIZER_PATH="checkpoints/char_tokenizer.json"
PROMPT="Hello"
MAX_LENGTH=100
TEMPERATURE=0.8

# Display help
show_help() {
    echo "Custom Language Model - All-in-One Tool"
    echo "========================================"
    echo ""
    echo "Usage: ./llm.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  setup               Set up the environment"
    echo "  train               Train a new model"
    echo "  generate            Generate text"
    echo "  test                Run tests"
    echo "  help                Show this help message"
    echo ""
    echo "Training options:"
    echo "  --data-dir DIR        Directory with training data (default: data/raw)"
    echo "  --tokenizer TYPE      Tokenizer type: 'char' or 'word' (default: char)"
    echo "  --epochs N            Number of training epochs (default: 10)"
    echo "  --d-model N           Model dimension (default: 64)"
    echo "  --layers N            Number of transformer layers (default: 2)"
    echo "  --heads N             Number of attention heads (default: 4)"
    echo "  --batch-size N        Batch size (default: 4)"
    echo ""
    echo "Generation options:"
    echo "  --model PATH          Path to model checkpoint (default: checkpoints/best_model.pt)"
    echo "  --tokenizer-path PATH Path to tokenizer file (default: checkpoints/char_tokenizer.json)"
    echo "  --prompt TEXT         Text prompt for generation (default: 'Hello')"
    echo "  --max-length N        Maximum length to generate (default: 100)"
    echo "  --temperature N       Sampling temperature (default: 0.8)"
    echo ""
    echo "Examples:"
    echo "  ./llm.sh setup                       # Set up the environment"
    echo "  ./llm.sh train --epochs 20 --d-model 128"
    echo "  ./llm.sh generate --prompt 'Once upon a time' --temperature 0.5"
    echo "  ./llm.sh test                        # Run tests"
}

# Setup function
setup() {
    echo "Setting up Custom LLM environment..."
    
    # Create directories
    mkdir -p checkpoints
    mkdir -p data/raw
    mkdir -p data/processed
    
    # Install dependencies
    pip install -r requirements.txt
    
    echo "Setup complete! You can now use the custom LLM."
    echo ""
    echo "To train a model with the sample data:"
    echo "./llm.sh train"
    echo ""
    echo "To explore the model in a notebook:"
    echo "jupyter notebook notebooks/model_exploration.ipynb"
}

# Parse command
if [ $# -gt 0 ]; then
    case "$1" in
        setup)
            MODE="setup"
            shift
            ;;
        train)
            MODE="train"
            shift
            ;;
        generate)
            MODE="generate"
            shift
            ;;
        test)
            MODE="test"
            shift
            ;;
        help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
fi

# Parse options
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --help)
            show_help
            exit 0
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift
            shift
            ;;
        --tokenizer)
            TOKENIZER_TYPE="$2"
            shift
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift
            shift
            ;;
        --d-model)
            D_MODEL="$2"
            shift
            shift
            ;;
        --layers)
            NUM_LAYERS="$2"
            shift
            shift
            ;;
        --heads)
            NUM_HEADS="$2"
            shift
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --model)
            MODEL_PATH="$2"
            shift
            shift
            ;;
        --tokenizer-path)
            TOKENIZER_PATH="$2"
            shift
            shift
            ;;
        --prompt)
            PROMPT="$2"
            shift
            shift
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift
            shift
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Execute the requested mode
case "$MODE" in
    setup)
        setup
        ;;
    train)
        echo "Training model with:"
        echo "  Data directory: $DATA_DIR"
        echo "  Tokenizer type: $TOKENIZER_TYPE"
        echo "  Epochs: $EPOCHS"
        echo "  Model dimension: $D_MODEL"
        echo "  Layers: $NUM_LAYERS"
        echo "  Attention heads: $NUM_HEADS"
        echo "  Batch size: $BATCH_SIZE"
        echo ""
        
        python3 src/main.py \
            --mode train \
            --data_dir "$DATA_DIR" \
            --tokenizer_type "$TOKENIZER_TYPE" \
            --epochs "$EPOCHS" \
            --d_model "$D_MODEL" \
            --num_layers "$NUM_LAYERS" \
            --num_heads "$NUM_HEADS" \
            --batch_size "$BATCH_SIZE"
        ;;
    generate)
        echo "Generating text with:"
        echo "  Model: $MODEL_PATH"
        echo "  Tokenizer: $TOKENIZER_PATH"
        echo "  Prompt: '$PROMPT'"
        echo "  Max length: $MAX_LENGTH"
        echo "  Temperature: $TEMPERATURE"
        echo ""
        
        # Check if model and tokenizer exist
        if [ ! -f "$MODEL_PATH" ]; then
            echo "Error: Model file not found at $MODEL_PATH"
            echo "Please train a model first or specify a valid model path with --model"
            exit 1
        fi

        if [ ! -f "$TOKENIZER_PATH" ]; then
            echo "Error: Tokenizer file not found at $TOKENIZER_PATH"
            echo "Please train a model first or specify a valid tokenizer path with --tokenizer-path"
            exit 1
        fi
        
        python3 src/main.py \
            --mode generate \
            --model_path "$MODEL_PATH" \
            --tokenizer_path "$TOKENIZER_PATH" \
            --prompt "$PROMPT" \
            --max_length "$MAX_LENGTH" \
            --temperature "$TEMPERATURE" \
            --d_model "$D_MODEL" \
            --num_layers "$NUM_LAYERS" \
            --num_heads "$NUM_HEADS"
        ;;
    test)
        echo "Running tests..."
        python3 -m unittest discover tests
        ;;
    help|*)
        show_help
        ;;
esac
