import os
import json
from collections import Counter
from typing import List, Dict, Optional, Union

class SimpleTokenizer:
    """
    A basic tokenizer that handles vocabulary creation and tokenization.
    This is a simplified version - for production, consider using established 
    tokenizers like BPE, WordPiece, or SentencePiece.
    """
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def train(self, texts: List[str]):
        """Build vocabulary from a list of texts"""
        # Count word frequencies
        counter = Counter()
        for text in texts:
            # Simple whitespace tokenization for demonstration
            words = text.lower().split()
            counter.update(words)
        
        # Select top words after special tokens
        num_special = len(self.special_tokens)
        most_common = counter.most_common(self.vocab_size - num_special)
        
        # Add to vocabulary
        for idx, (word, _) in enumerate(most_common):
            token_id = idx + num_special
            self.token_to_id[word] = token_id
            self.id_to_token[token_id] = word
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to token IDs"""
        # Simple whitespace tokenization
        words = text.lower().split()
        
        ids = []
        if add_special_tokens:
            ids.append(self.special_tokens["<bos>"])
        
        for word in words:
            if word in self.token_to_id:
                ids.append(self.token_to_id[word])
            else:
                ids.append(self.special_tokens["<unk>"])
        
        if add_special_tokens:
            ids.append(self.special_tokens["<eos>"])
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text"""
        special_ids = set(self.special_tokens.values()) if skip_special_tokens else set()
        
        words = []
        for idx in ids:
            if idx in special_ids:
                continue
            if idx in self.id_to_token:
                words.append(self.id_to_token[idx])
            else:
                words.append(self.id_to_token[self.special_tokens["<unk>"]])
        
        return " ".join(words)
    
    def save(self, path: str):
        """Save tokenizer vocabulary to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "token_to_id": self.token_to_id,
                "special_tokens": self.special_tokens
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        """Load tokenizer vocabulary from disk"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.special_tokens = data["special_tokens"]
        
        # Rebuild id_to_token mapping
        tokenizer.id_to_token = {int(idx): token for token, idx in tokenizer.token_to_id.items()}
        
        return tokenizer


class CharacterTokenizer:
    """
    A character-level tokenizer - simpler than word-level but often effective for small models
    """
    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def train(self, texts: List[str]):
        """Build vocabulary from a list of texts"""
        # Get all unique characters
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)
        
        print(f"Found {len(unique_chars)} unique characters")
        
        # Add to vocabulary after special tokens
        num_special = len(self.special_tokens)
        for idx, char in enumerate(sorted(unique_chars)):
            token_id = idx + num_special
            self.token_to_id[char] = token_id
            self.id_to_token[token_id] = char
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to token IDs"""
        ids = []
        if add_special_tokens:
            ids.append(self.special_tokens["<bos>"])
        
        for char in text:
            if char in self.token_to_id:
                ids.append(self.token_to_id[char])
            else:
                ids.append(self.special_tokens["<unk>"])
        
        if add_special_tokens:
            ids.append(self.special_tokens["<eos>"])
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text"""
        special_ids = set(self.special_tokens.values()) if skip_special_tokens else set()
        
        chars = []
        for idx in ids:
            if idx in special_ids:
                continue
            if idx in self.id_to_token:
                chars.append(self.id_to_token[idx])
            else:
                chars.append(self.id_to_token[self.special_tokens["<unk>"]])
        
        return "".join(chars)
    
    def save(self, path: str):
        """Save tokenizer vocabulary to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                "token_to_id": self.token_to_id,
                "special_tokens": self.special_tokens
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "CharacterTokenizer":
        """Load tokenizer vocabulary from disk"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.special_tokens = data["special_tokens"]
        
        # Rebuild id_to_token mapping
        tokenizer.id_to_token = {int(idx): token for token, idx in tokenizer.token_to_id.items()}
        
        return tokenizer
