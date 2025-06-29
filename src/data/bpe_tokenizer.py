import os
import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional

class BPETokenizer:
    """
    A simple implementation of Byte-Pair Encoding (BPE) tokenizer.
    BPE is a subword tokenization algorithm that iteratively merges the most frequent
    pairs of characters or character sequences.
    """
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
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
    
    def _get_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent pairs in the vocabulary"""
        pairs = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs
    
    def _merge_vocab(self, words: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        """Merge all occurrences of a pair in the vocabulary"""
        first, second = pair
        new_words = []
        
        for word in words:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        
        return new_words
    
    def train(self, texts: List[str], min_frequency: int = 2, num_merges: Optional[int] = None):
        """
        Train the BPE tokenizer on a list of texts.
        
        Args:
            texts: List of text strings to train on
            min_frequency: Minimum frequency for a token to be included
            num_merges: Number of merge operations to perform (if None, use vocab_size)
        """
        print(f"Training BPE tokenizer on {len(texts)} texts...")
        
        # Start with character-level tokens
        words = []
        for text in texts:
            # Simple preprocessing: lowercase and split on whitespace
            for word in re.findall(r'\S+|\s+', text):
                words.append(list(word))
        
        # Count initial vocabulary
        vocab = Counter()
        for word in words:
            vocab.update(word)
        
        # Filter by frequency
        vocab = {token: count for token, count in vocab.items() if count >= min_frequency}
        
        # Add all characters to vocabulary
        num_special = len(self.special_tokens)
        for idx, (token, _) in enumerate(sorted(vocab.items())):
            token_id = idx + num_special
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        
        # Determine number of merges
        if num_merges is None:
            num_merges = min(self.vocab_size - len(self.token_to_id), 10000)
        
        # Perform BPE merges
        for i in range(num_merges):
            if len(self.token_to_id) >= self.vocab_size:
                break
                
            # Get pair statistics
            pairs = self._get_stats(words)
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_frequency:
                break
                
            # Create new token for the pair
            new_token = ''.join(best_pair)
            token_id = len(self.token_to_id)
            self.token_to_id[new_token] = token_id
            self.id_to_token[token_id] = new_token
            
            # Add merge to list of merges
            self.merges.append(best_pair)
            
            # Merge the pair in all words
            words = self._merge_vocab(words, best_pair)
            
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1} merges. Vocabulary size: {len(self.token_to_id)}")
        
        print(f"BPE training complete. Final vocabulary size: {len(self.token_to_id)}")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using learned BPE merges"""
        # Start with characters
        chars = list(word)
        
        # Apply merges in order
        for first, second in self.merges:
            i = 0
            while i < len(chars) - 1:
                if chars[i] == first and chars[i + 1] == second:
                    chars[i] = first + second
                    chars.pop(i + 1)
                else:
                    i += 1
        
        return chars
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to token IDs"""
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.special_tokens["<bos>"])
        
        # Tokenize each word
        for word in re.findall(r'\S+|\s+', text):
            for token in self._tokenize_word(word):
                if token in self.token_to_id:
                    tokens.append(self.token_to_id[token])
                else:
                    tokens.append(self.special_tokens["<unk>"])
        
        if add_special_tokens:
            tokens.append(self.special_tokens["<eos>"])
        
        return tokens
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text"""
        special_ids = set(self.special_tokens.values()) if skip_special_tokens else set()
        
        tokens = []
        for idx in ids:
            if idx in special_ids:
                continue
            if idx in self.id_to_token:
                tokens.append(self.id_to_token[idx])
            else:
                tokens.append(self.id_to_token[self.special_tokens["<unk>"]])
        
        return ''.join(tokens)
    
    def save(self, path: str):
        """Save tokenizer vocabulary and merges to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "token_to_id": self.token_to_id,
                "merges": self.merges,
                "special_tokens": self.special_tokens
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer vocabulary and merges from disk"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.merges = [tuple(merge) for merge in data["merges"]]
        tokenizer.special_tokens = data["special_tokens"]
        
        # Rebuild id_to_token mapping
        tokenizer.id_to_token = {int(idx): token for token, idx in tokenizer.token_to_id.items()}
        
        return tokenizer
