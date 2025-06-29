import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import json

from model.transformer import CustomLanguageModel
from data.tokenizer import SimpleTokenizer, CharacterTokenizer
from data.bpe_tokenizer import BPETokenizer
from data.dataset import load_and_preprocess_text, create_dataloaders
from training.trainer import Trainer
from evaluation.metrics import calculate_perplexity, calculate_accuracy, evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train or use a custom language model")
    parser.add_argument("--mode", type=str, choices=["train", "generate", "evaluate"], default="train",
                        help="Whether to train the model, generate text, or evaluate")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/raw",
                        help="Directory containing training data")
    parser.add_argument("--tokenizer_type", type=str, choices=["char", "word", "bpe"], default="char",
                        help="Type of tokenizer to use")
    parser.add_argument("--vocab_size", type=int, default=1000,
                        help="Vocabulary size for tokenizers that support it")
    parser.add_argument("--block_size", type=int, default=128,
                        help="Context size for the model")
    
    # Model arguments
    parser.add_argument("--d_model", type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=1024,
                        help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use Weights & Biases for logging")
    
    # Generation arguments
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a trained model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to a saved tokenizer")
    parser.add_argument("--prompt", type=str, default="",
                        help="Prompt for text generation")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature for generation")
    parser.add_argument("--use_beam", action="store_true",
                        help="Use beam search for generation")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size for beam search")
    
    # Evaluation arguments
    parser.add_argument("--eval_data", type=str, default=None,
                        help="Path to evaluation data file")
    parser.add_argument("--eval_samples", type=int, default=5,
                        help="Number of samples to use for evaluation")
    
    return parser.parse_args()

def train(args):
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Get data files
    data_files = glob.glob(os.path.join(args.data_dir, "*.txt"))
    if not data_files:
        raise ValueError(f"No text files found in {args.data_dir}")
    
    print(f"Found {len(data_files)} data files")
    
    # Create tokenizer
    if args.tokenizer_type == "char":
        tokenizer = CharacterTokenizer()
    elif args.tokenizer_type == "bpe":
        tokenizer = BPETokenizer(vocab_size=args.vocab_size if hasattr(args, 'vocab_size') else 1000)
    else:
        tokenizer = SimpleTokenizer()
    
    # Load and preprocess data
    print("Loading and tokenizing data...")
    tokenized_texts = load_and_preprocess_text(data_files, tokenizer)
    
    # Check if we have data
    if not tokenized_texts or all(len(text) == 0 for text in tokenized_texts):
        raise ValueError("No data found in the provided files or tokenization failed")
    
    print(f"Tokenized {len(tokenized_texts)} texts with a total of {sum(len(text) for text in tokenized_texts)} tokens")
    
    # Save tokenizer
    tokenizer_path = os.path.join(args.checkpoint_dir, f"{args.tokenizer_type}_tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        tokenized_texts,
        block_size=args.block_size,
        batch_size=args.batch_size
    )
    
    # Create model
    vocab_size = len(tokenizer.token_to_id)
    print(f"Vocabulary size: {vocab_size}")
    
    model = CustomLanguageModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_length=args.block_size,
        dropout=args.dropout
    )
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb
    )
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    trainer.train(num_epochs=args.epochs)
    
    print("Training complete!")

def generate(args):
    # Load tokenizer
    if args.tokenizer_path is None:
        raise ValueError("Must provide --tokenizer_path for generation")
    
    if "char" in args.tokenizer_path:
        tokenizer = CharacterTokenizer.load(args.tokenizer_path)
    else:
        tokenizer = SimpleTokenizer.load(args.tokenizer_path)
    
    # Load model
    if args.model_path is None:
        raise ValueError("Must provide --model_path for generation")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint to get model parameters
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Create model with the same parameters as during training
    vocab_size = len(tokenizer.token_to_id)
    model = CustomLanguageModel(
        vocab_size=vocab_size,
        d_model=64,  # Use the same values as during training
        num_heads=4,
        num_layers=2,
        d_ff=1024,
        max_seq_length=args.block_size,
        dropout=0.1
    )
    
    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Prepare prompt
    if not args.prompt:
        args.prompt = input("Enter a prompt: ")
    
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long).to(device)
    
    # Generate text
    print(f"Generating text with prompt: '{args.prompt}'")
    
    if args.use_beam:
        print(f"Using beam search with beam size: {args.beam_size}")
        generated_ids = model.generate_beam(
            prompt_ids, 
            max_length=args.max_length,
            beam_size=args.beam_size
        )
    else:
        print(f"Using sampling with temperature: {args.temperature}")
        generated_ids = model.generate(
            prompt_ids, 
            max_length=args.max_length, 
            temperature=args.temperature
        )
    
    # Decode and print
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"\nGenerated text:\n{generated_text}")

def evaluate(args):
    # Load tokenizer
    if args.tokenizer_path is None:
        raise ValueError("Must provide --tokenizer_path for evaluation")
    
    if "char" in args.tokenizer_path:
        tokenizer = CharacterTokenizer.load(args.tokenizer_path)
    elif "bpe" in args.tokenizer_path:
        tokenizer = BPETokenizer.load(args.tokenizer_path)
    else:
        tokenizer = SimpleTokenizer.load(args.tokenizer_path)
    
    # Load model
    if args.model_path is None:
        raise ValueError("Must provide --model_path for evaluation")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    vocab_size = len(tokenizer.token_to_id)
    model = CustomLanguageModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_length=args.block_size,
        dropout=args.dropout
    )
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Load evaluation data
    if args.eval_data:
        with open(args.eval_data, 'r', encoding='utf-8') as f:
            eval_text = f.read()
    else:
        # Use sample data if no evaluation data is provided
        data_files = glob.glob(os.path.join("data/raw", "*.txt"))
        if not data_files:
            raise ValueError("No evaluation data provided and no text files found in data/raw")
        
        with open(data_files[0], 'r', encoding='utf-8') as f:
            eval_text = f.read()
    
    # Split into paragraphs or sentences for evaluation
    import re
    eval_samples = re.split(r'\n\n|\.\s+', eval_text)
    eval_samples = [s.strip() for s in eval_samples if len(s.strip()) > 50]
    
    # Limit number of samples
    if args.eval_samples > 0 and args.eval_samples < len(eval_samples):
        eval_samples = eval_samples[:args.eval_samples]
    
    print(f"Evaluating model on {len(eval_samples)} text samples...")
    
    # Run evaluation
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_texts=eval_samples,
        device=device,
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    # Calculate perplexity if possible
    try:
        # Create a small dataset for perplexity calculation
        from data.dataset import TextDataset
        tokenized_texts = [tokenizer.encode(text) for text in eval_samples]
        dataset = TextDataset(tokenized_texts, block_size=args.block_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        perplexity = calculate_perplexity(model, dataloader, device)
        metrics["perplexity"] = perplexity
    except Exception as e:
        print(f"Could not calculate perplexity: {e}")
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results to file
    results_path = os.path.join(os.path.dirname(args.model_path), "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return metrics

def main():
    args = parse_args()
    
    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        generate(args)

if __name__ == "__main__":
    main()
