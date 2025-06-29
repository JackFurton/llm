import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from typing import List, Dict, Optional, Union, Tuple

class TextDataset(Dataset):
    """
    Dataset for language modeling tasks
    """
    def __init__(
        self, 
        tokenized_texts: List[List[int]], 
        block_size: int = 128,
        stride: Optional[int] = None
    ):
        self.examples = []
        stride = stride if stride is not None else block_size
        
        # Create training examples by sliding a window over the tokenized texts
        for tokens in tokenized_texts:
            # Handle case where tokens is shorter than block_size
            if len(tokens) <= block_size:
                # Pad if necessary
                if len(tokens) < block_size + 1:  # +1 for the label
                    padded = tokens + [0] * (block_size + 1 - len(tokens))
                    self.examples.append(padded)
            else:
                # Normal sliding window approach
                for i in range(0, len(tokens) - block_size, stride):
                    # Extract a block of tokens
                    input_chunk = tokens[i:i + block_size + 1]  # +1 for the label
                    
                    # If the chunk is smaller than block_size+1, pad it
                    if len(input_chunk) < block_size + 1:
                        input_chunk = input_chunk + [0] * (block_size + 1 - len(input_chunk))
                    
                    self.examples.append(input_chunk)
        
        # Ensure we have at least one example
        if not self.examples and tokenized_texts:
            # Create at least one example from the first text
            tokens = tokenized_texts[0]
            # Repeat the tokens if necessary to reach block_size+1
            while len(tokens) < block_size + 1:
                tokens = tokens + tokens
            self.examples.append(tokens[:block_size + 1])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # For causal language modeling:
        # - input_ids are the original tokens
        # - labels are the same tokens shifted by 1 (predict next token)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }


def create_dataloaders(
    tokenized_texts: List[List[int]],
    train_ratio: float = 0.9,
    block_size: int = 128,
    batch_size: int = 16,
    stride: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from tokenized texts
    """
    if not tokenized_texts:
        raise ValueError("No tokenized texts provided")
    
    # Split data into train and validation
    split_idx = max(1, int(len(tokenized_texts) * train_ratio))
    train_texts = tokenized_texts[:split_idx]
    val_texts = tokenized_texts[split_idx:] if split_idx < len(tokenized_texts) else [tokenized_texts[0]]
    
    # Create datasets
    train_dataset = TextDataset(train_texts, block_size=block_size, stride=stride)
    val_dataset = TextDataset(val_texts, block_size=block_size, stride=block_size)  # No overlap for validation
    
    # Check if datasets are empty
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check your data and block_size.")
    
    if len(val_dataset) == 0:
        print("Warning: Validation dataset is empty. Using a copy of the training dataset.")
        val_dataset = train_dataset
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_dataloader, val_dataloader


def load_and_preprocess_text(
    file_paths: List[str],
    tokenizer,
    max_length: Optional[int] = None
) -> List[List[int]]:
    """
    Load text files, tokenize them, and prepare for language modeling
    """
    tokenized_texts = []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Loaded {len(text)} characters from {file_path}")
        
        # Train tokenizer if it's the first file
        if len(tokenized_texts) == 0:
            tokenizer.train([text])
            print(f"Tokenizer vocabulary size: {len(tokenizer.token_to_id)}")
        
        # Tokenize the text
        tokens = tokenizer.encode(text)
        print(f"Encoded to {len(tokens)} tokens")
        
        # Optionally truncate
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
            print(f"Truncated to {len(tokens)} tokens")
        
        tokenized_texts.append(tokens)
    
    return tokenized_texts


class MemoryEfficientTextDataset(Dataset):
    """
    Memory-efficient dataset that loads and tokenizes text on-the-fly
    Useful for very large datasets that don't fit in memory
    """
    def __init__(
        self,
        file_paths: List[str],
        tokenizer,
        block_size: int = 128
    ):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Calculate total number of blocks
        self.file_sizes = []
        self.cumulative_blocks = [0]
        
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            tokens = len(self.tokenizer.encode(text))
            blocks = max(1, (tokens - 1) // block_size)
            self.file_sizes.append(blocks)
            self.cumulative_blocks.append(self.cumulative_blocks[-1] + blocks)
    
    def __len__(self):
        return self.cumulative_blocks[-1]
    
    def __getitem__(self, idx):
        # Find which file this index belongs to
        file_idx = 0
        while idx >= self.cumulative_blocks[file_idx + 1]:
            file_idx += 1
        
        # Calculate position within the file
        local_idx = idx - self.cumulative_blocks[file_idx]
        
        # Load and tokenize the file
        with open(self.file_paths[file_idx], 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = self.tokenizer.encode(text)
        
        # Extract the block
        start_pos = local_idx * self.block_size
        end_pos = min(start_pos + self.block_size + 1, len(tokens))  # +1 for the label
        block = tokens[start_pos:end_pos]
        
        # Pad if necessary
        if len(block) < self.block_size + 1:
            block = block + [0] * (self.block_size + 1 - len(block))
        
        # Prepare input and labels
        input_ids = torch.tensor(block[:-1], dtype=torch.long)
        labels = torch.tensor(block[1:], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }
