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
PROCESSED_DIR="data/processed"
CURATED_DIR="data/curated"
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
USE_BEAM=false
BEAM_SIZE=5
VOCAB_SIZE=1000
EVAL_DATA=""
EVAL_SAMPLES=5
COLLECT_SOURCES="all"
COLLECT_QUERY=""
COLLECT_LIMIT=5
COLLECT_SUBREDDIT=""
COLLECT_NEWS_FEED=""
PREPROCESS_LANGUAGE="en"
PREPROCESS_NORMALIZE=false
PREPROCESS_CLEAN_HTML=false
PREPROCESS_CLEAN_MARKDOWN=false
PREPROCESS_CLEAN_WHITESPACE=false
PREPROCESS_AUGMENT=false
WEB_HOST="127.0.0.1"
WEB_PORT=5000
WEB_DEBUG=false

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
    echo "  evaluate            Evaluate model performance"
    echo "  collect             Collect training data"
    echo "  preprocess          Preprocess training data"
    echo "  curate              Launch web interface for data curation"
    echo "  test                Run tests"
    echo "  help                Show this help message"
    echo ""
    echo "Training options:"
    echo "  --data-dir DIR        Directory with training data (default: data/raw)"
    echo "  --processed-dir DIR   Directory with processed data (default: data/processed)"
    echo "  --curated-dir DIR     Directory with curated data (default: data/curated)"
    echo "  --tokenizer TYPE      Tokenizer type: 'char', 'word', or 'bpe' (default: char)"
    echo "  --vocab-size N        Vocabulary size for BPE tokenizer (default: 1000)"
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
    echo "  --beam                Use beam search instead of sampling"
    echo "  --beam-size N         Beam size for beam search (default: 5)"
    echo ""
    echo "Evaluation options:"
    echo "  --eval-data PATH      Path to evaluation data file"
    echo "  --eval-samples N      Number of samples to use for evaluation (default: 5)"
    echo ""
    echo "Data collection options:"
    echo "  --sources LIST        Data sources to collect from: 'wikipedia', 'gutenberg', 'news', 'reddit', 'all' (default: all)"
    echo "  --query TEXT          Search query for data collection"
    echo "  --limit N             Maximum number of items to collect per source (default: 5)"
    echo "  --subreddit NAME      Specific subreddit to collect from (for Reddit source)"
    echo "  --news-feed URL       Specific news feed URL to collect from (for News source)"
    echo ""
    echo "Data preprocessing options:"
    echo "  --language LANG       Language filter (default: en)"
    echo "  --no-language-filter  Disable language filtering"
    echo "  --no-content-filter   Disable content filtering"
    echo "  --no-quality-filter   Disable quality filtering"
    echo "  --no-duplicate-filter Disable duplicate filtering"
    echo "  --normalize           Apply text normalization"
    echo "  --clean-html          Clean HTML tags and entities"
    echo "  --clean-markdown      Clean Markdown formatting"
    echo "  --clean-whitespace    Clean and normalize whitespace"
    echo "  --augment             Apply text augmentation"
    echo "  --synonym-replace     Replace words with synonyms"
    echo "  --back-translate      Apply back-translation augmentation"
    echo ""
    echo "Web interface options:"
    echo "  --host HOST           Host to run the web server on (default: 127.0.0.1)"
    echo "  --port PORT           Port to run the web server on (default: 5000)"
    echo "  --debug               Run the web server in debug mode"
    echo ""
    echo "Examples:"
    echo "  ./llm.sh setup                       # Set up the environment"
    echo "  ./llm.sh collect --sources wikipedia --query 'machine learning' --limit 10"
    echo "  ./llm.sh preprocess --normalize --clean-html --clean-whitespace"
    echo "  ./llm.sh curate                      # Launch web interface for data curation"
    echo "  ./llm.sh train --tokenizer bpe --vocab-size 5000 --epochs 20"
    echo "  ./llm.sh generate --prompt 'Once upon a time' --temperature 0.5"
    echo "  ./llm.sh evaluate --model checkpoints/best_model.pt"
    echo "  ./llm.sh test                        # Run tests"
}

# Setup function
setup() {
    echo "Setting up Custom LLM environment..."
    
    # Create directories
    mkdir -p checkpoints
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p data/curated
    
    # Install dependencies
    pip install -r requirements.txt
    
    echo "Setup complete! You can now use the custom LLM."
    echo ""
    echo "To collect training data:"
    echo "./llm.sh collect --sources wikipedia --query 'artificial intelligence'"
    echo ""
    echo "To preprocess the collected data:"
    echo "./llm.sh preprocess --normalize --clean-html --clean-whitespace"
    echo ""
    echo "To curate the processed data:"
    echo "./llm.sh curate"
    echo ""
    echo "To train a model with the curated data:"
    echo "./llm.sh train --curated-dir data/curated"
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
        evaluate)
            MODE="evaluate"
            shift
            ;;
        collect)
            MODE="collect"
            shift
            ;;
        preprocess)
            MODE="preprocess"
            shift
            ;;
        curate)
            MODE="curate"
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
        --processed-dir)
            PROCESSED_DIR="$2"
            shift
            shift
            ;;
        --curated-dir)
            CURATED_DIR="$2"
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
        --beam)
            USE_BEAM=true
            shift
            ;;
        --beam-size)
            BEAM_SIZE="$2"
            shift
            shift
            ;;
        --vocab-size)
            VOCAB_SIZE="$2"
            shift
            shift
            ;;
        --eval-data)
            EVAL_DATA="$2"
            shift
            shift
            ;;
        --eval-samples)
            EVAL_SAMPLES="$2"
            shift
            shift
            ;;
        --sources)
            COLLECT_SOURCES="$2"
            shift
            shift
            ;;
        --query)
            COLLECT_QUERY="$2"
            shift
            shift
            ;;
        --limit)
            COLLECT_LIMIT="$2"
            shift
            shift
            ;;
        --subreddit)
            COLLECT_SUBREDDIT="$2"
            shift
            shift
            ;;
        --news-feed)
            COLLECT_NEWS_FEED="$2"
            shift
            shift
            ;;
        --language)
            PREPROCESS_LANGUAGE="$2"
            shift
            shift
            ;;
        --no-language-filter)
            PREPROCESS_NO_LANGUAGE_FILTER="--no-language-filter"
            shift
            ;;
        --no-content-filter)
            PREPROCESS_NO_CONTENT_FILTER="--no-content-filter"
            shift
            ;;
        --no-quality-filter)
            PREPROCESS_NO_QUALITY_FILTER="--no-quality-filter"
            shift
            ;;
        --no-duplicate-filter)
            PREPROCESS_NO_DUPLICATE_FILTER="--no-duplicate-filter"
            shift
            ;;
        --normalize)
            PREPROCESS_NORMALIZE="--normalize"
            shift
            ;;
        --clean-html)
            PREPROCESS_CLEAN_HTML="--clean-html"
            shift
            ;;
        --clean-markdown)
            PREPROCESS_CLEAN_MARKDOWN="--clean-markdown"
            shift
            ;;
        --clean-whitespace)
            PREPROCESS_CLEAN_WHITESPACE="--clean-whitespace"
            shift
            ;;
        --augment)
            PREPROCESS_AUGMENT="--augment"
            shift
            ;;
        --synonym-replace)
            PREPROCESS_SYNONYM_REPLACE="--synonym-replace"
            shift
            ;;
        --back-translate)
            PREPROCESS_BACK_TRANSLATE="--back-translate"
            shift
            ;;
        --host)
            WEB_HOST="$2"
            shift
            shift
            ;;
        --port)
            WEB_PORT="$2"
            shift
            shift
            ;;
        --debug)
            WEB_DEBUG=true
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
        echo "  Processed directory: $PROCESSED_DIR"
        echo "  Curated directory: $CURATED_DIR"
        echo "  Tokenizer type: $TOKENIZER_TYPE"
        echo "  Epochs: $EPOCHS"
        echo "  Model dimension: $D_MODEL"
        echo "  Layers: $NUM_LAYERS"
        echo "  Attention heads: $NUM_HEADS"
        echo "  Batch size: $BATCH_SIZE"
        echo ""
        
        # Use curated data if available, otherwise use processed data if available, otherwise use raw data
        TRAIN_DATA_DIR="$DATA_DIR"
        if [ -d "$CURATED_DIR" ] && [ "$(ls -A $CURATED_DIR)" ]; then
            echo "Using curated data from $CURATED_DIR"
            TRAIN_DATA_DIR="$CURATED_DIR"
        elif [ -d "$PROCESSED_DIR" ] && [ "$(ls -A $PROCESSED_DIR)" ]; then
            echo "Using processed data from $PROCESSED_DIR"
            TRAIN_DATA_DIR="$PROCESSED_DIR"
        fi
        
        python3 src/main.py \
            --mode train \
            --data_dir "$TRAIN_DATA_DIR" \
            --tokenizer_type "$TOKENIZER_TYPE" \
            --vocab_size "$VOCAB_SIZE" \
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
        if [ "$USE_BEAM" = true ]; then
            echo "  Using beam search with beam size: $BEAM_SIZE"
        else
            echo "  Using sampling"
        fi
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
        
        # Add beam search option if enabled
        BEAM_OPTION=""
        if [ "$USE_BEAM" = true ]; then
            BEAM_OPTION="--use_beam --beam_size $BEAM_SIZE"
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
            --num_heads "$NUM_HEADS" \
            $BEAM_OPTION
        ;;
    evaluate)
        echo "Evaluating model with:"
        echo "  Model: $MODEL_PATH"
        echo "  Tokenizer: $TOKENIZER_PATH"
        echo "  Samples: $EVAL_SAMPLES"
        if [ -n "$EVAL_DATA" ]; then
            echo "  Evaluation data: $EVAL_DATA"
        else
            echo "  Using default data"
        fi
        echo ""
        
        EVAL_DATA_OPTION=""
        if [ -n "$EVAL_DATA" ]; then
            EVAL_DATA_OPTION="--eval_data $EVAL_DATA"
        fi
        
        python3 src/main.py \
            --mode evaluate \
            --model_path "$MODEL_PATH" \
            --tokenizer_path "$TOKENIZER_PATH" \
            --eval_samples "$EVAL_SAMPLES" \
            --d_model "$D_MODEL" \
            --num_layers "$NUM_LAYERS" \
            --num_heads "$NUM_HEADS" \
            --max_length "$MAX_LENGTH" \
            --temperature "$TEMPERATURE" \
            $EVAL_DATA_OPTION
        ;;
    collect)
        echo "Collecting training data:"
        echo "  Sources: $COLLECT_SOURCES"
        echo "  Query: '$COLLECT_QUERY'"
        echo "  Limit per source: $COLLECT_LIMIT"
        echo "  Output directory: $DATA_DIR"
        if [ -n "$COLLECT_SUBREDDIT" ]; then
            echo "  Subreddit: $COLLECT_SUBREDDIT"
        fi
        if [ -n "$COLLECT_NEWS_FEED" ]; then
            echo "  News feed: $COLLECT_NEWS_FEED"
        fi
        echo ""
        
        # Build command with options
        COLLECT_CMD="python3 src/data_collection/collect.py --output-dir $DATA_DIR --limit $COLLECT_LIMIT"
        
        if [ -n "$COLLECT_SOURCES" ] && [ "$COLLECT_SOURCES" != "all" ]; then
            COLLECT_CMD="$COLLECT_CMD --sources $COLLECT_SOURCES"
        fi
        
        if [ -n "$COLLECT_QUERY" ]; then
            COLLECT_CMD="$COLLECT_CMD --query \"$COLLECT_QUERY\""
        fi
        
        if [ -n "$COLLECT_SUBREDDIT" ]; then
            COLLECT_CMD="$COLLECT_CMD --subreddit \"$COLLECT_SUBREDDIT\""
        fi
        
        if [ -n "$COLLECT_NEWS_FEED" ]; then
            COLLECT_CMD="$COLLECT_CMD --news-feed \"$COLLECT_NEWS_FEED\""
        fi
        
        # Execute the command
        eval $COLLECT_CMD
        ;;
    preprocess)
        echo "Preprocessing training data:"
        echo "  Input directory: $DATA_DIR"
        echo "  Output directory: $PROCESSED_DIR"
        echo "  Language: $PREPROCESS_LANGUAGE"
        
        if [ "$PREPROCESS_NORMALIZE" = "--normalize" ]; then
            echo "  Applying text normalization"
        fi
        if [ "$PREPROCESS_CLEAN_HTML" = "--clean-html" ]; then
            echo "  Cleaning HTML"
        fi
        if [ "$PREPROCESS_CLEAN_MARKDOWN" = "--clean-markdown" ]; then
            echo "  Cleaning Markdown"
        fi
        if [ "$PREPROCESS_CLEAN_WHITESPACE" = "--clean-whitespace" ]; then
            echo "  Cleaning whitespace"
        fi
        if [ "$PREPROCESS_AUGMENT" = "--augment" ]; then
            echo "  Applying text augmentation"
        fi
        if [ "$PREPROCESS_SYNONYM_REPLACE" = "--synonym-replace" ]; then
            echo "  Replacing words with synonyms"
        fi
        if [ "$PREPROCESS_BACK_TRANSLATE" = "--back-translate" ]; then
            echo "  Applying back-translation"
        fi
        echo ""
        
        # Build command with options
        PREPROCESS_CMD="python3 src/data_preprocessing/preprocess.py --input-dir $DATA_DIR --output-dir $PROCESSED_DIR --language $PREPROCESS_LANGUAGE"
        
        # Add filter options
        if [ -n "$PREPROCESS_NO_LANGUAGE_FILTER" ]; then
            PREPROCESS_CMD="$PREPROCESS_CMD $PREPROCESS_NO_LANGUAGE_FILTER"
        fi
        if [ -n "$PREPROCESS_NO_CONTENT_FILTER" ]; then
            PREPROCESS_CMD="$PREPROCESS_CMD $PREPROCESS_NO_CONTENT_FILTER"
        fi
        if [ -n "$PREPROCESS_NO_QUALITY_FILTER" ]; then
            PREPROCESS_CMD="$PREPROCESS_CMD $PREPROCESS_NO_QUALITY_FILTER"
        fi
        if [ -n "$PREPROCESS_NO_DUPLICATE_FILTER" ]; then
            PREPROCESS_CMD="$PREPROCESS_CMD $PREPROCESS_NO_DUPLICATE_FILTER"
        fi
        
        # Add normalizer options
        if [ "$PREPROCESS_NORMALIZE" = "--normalize" ]; then
            PREPROCESS_CMD="$PREPROCESS_CMD --normalize"
        fi
        if [ "$PREPROCESS_CLEAN_HTML" = "--clean-html" ]; then
            PREPROCESS_CMD="$PREPROCESS_CMD --clean-html"
        fi
        if [ "$PREPROCESS_CLEAN_MARKDOWN" = "--clean-markdown" ]; then
            PREPROCESS_CMD="$PREPROCESS_CMD --clean-markdown"
        fi
        if [ "$PREPROCESS_CLEAN_WHITESPACE" = "--clean-whitespace" ]; then
            PREPROCESS_CMD="$PREPROCESS_CMD --clean-whitespace"
        fi
        
        # Add augmenter options
        if [ "$PREPROCESS_AUGMENT" = "--augment" ]; then
            PREPROCESS_CMD="$PREPROCESS_CMD --augment"
        fi
        if [ "$PREPROCESS_SYNONYM_REPLACE" = "--synonym-replace" ]; then
            PREPROCESS_CMD="$PREPROCESS_CMD --synonym-replace"
        fi
        if [ "$PREPROCESS_BACK_TRANSLATE" = "--back-translate" ]; then
            PREPROCESS_CMD="$PREPROCESS_CMD --back-translate"
        fi
        
        # Execute the command
        eval $PREPROCESS_CMD
        ;;
    curate)
        echo "Launching data curation web interface:"
        echo "  Host: $WEB_HOST"
        echo "  Port: $WEB_PORT"
        echo "  Raw data directory: $DATA_DIR"
        echo "  Processed data directory: $PROCESSED_DIR"
        echo "  Curated data directory: $CURATED_DIR"
        if [ "$WEB_DEBUG" = true ]; then
            echo "  Debug mode: enabled"
        fi
        echo ""
        echo "Open your browser and navigate to http://$WEB_HOST:$WEB_PORT"
        echo "Press Ctrl+C to stop the server"
        echo ""
        
        # Build command with options
        CURATE_CMD="python3 src/web_interface/run.py --host $WEB_HOST --port $WEB_PORT --raw-dir $DATA_DIR --processed-dir $PROCESSED_DIR --curated-dir $CURATED_DIR"
        
        if [ "$WEB_DEBUG" = true ]; then
            CURATE_CMD="$CURATE_CMD --debug"
        fi
        
        # Execute the command
        eval $CURATE_CMD
        ;;
    test)
        echo "Running tests..."
        python3 -m unittest discover tests
        ;;
    help|*)
        show_help
        ;;
esac
