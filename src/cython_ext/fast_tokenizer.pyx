# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
"""
Optimized Cython implementation of the character tokenizer for improved performance.
"""
import json
import os
from collections import Counter
from typing import List, Dict, Optional, Union, Set
from libc.string cimport strlen

cdef class FastCharacterTokenizer:
    """
    A fast character-level tokenizer implemented in Cython with optimizations
    """
    cdef public dict token_to_id
    cdef public dict id_to_token
    cdef public dict special_tokens
    cdef int unk_id, bos_id, eos_id, pad_id
    
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        
        # Cache special token IDs for faster access
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def train(self, list texts):
        """Build vocabulary from a list of texts"""
        # Get all unique characters
        cdef set unique_chars = set()
        cdef str text
        cdef str char
        cdef int i, j, text_len
        
        for text in texts:
            text_len = len(text)
            for i in range(text_len):
                unique_chars.add(text[i])
        
        print(f"Found {len(unique_chars)} unique characters")
        
        # Add to vocabulary after special tokens
        cdef int num_special = len(self.special_tokens)
        cdef int token_id
        cdef int idx
        cdef list sorted_chars = sorted(unique_chars)
        
        for idx, char in enumerate(sorted_chars):
            token_id = idx + num_special
            self.token_to_id[char] = token_id
            self.id_to_token[token_id] = char
    
    cpdef list encode(self, str text, bint add_special_tokens=True):
        """Convert text to token IDs"""
        cdef list ids = []
        cdef int i, text_len = len(text)
        cdef str char
        
        if add_special_tokens:
            ids.append(self.bos_id)
        
        for i in range(text_len):
            char = text[i]
            if char in self.token_to_id:
                ids.append(self.token_to_id[char])
            else:
                ids.append(self.unk_id)
        
        if add_special_tokens:
            ids.append(self.eos_id)
        
        return ids
    
    cpdef str decode(self, list ids, bint skip_special_tokens=True):
        """Convert token IDs back to text"""
        cdef set special_ids
        cdef list chars = []
        cdef int i, ids_len = len(ids), idx
        
        if skip_special_tokens:
            special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        else:
            special_ids = set()
        
        for i in range(ids_len):
            idx = ids[i]
            if idx in special_ids:
                continue
            if idx in self.id_to_token:
                chars.append(self.id_to_token[idx])
            else:
                chars.append(self.id_to_token[self.unk_id])
        
        return "".join(chars)
    
    def save(self, str path):
        """Save tokenizer vocabulary to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                "token_to_id": self.token_to_id,
                "special_tokens": self.special_tokens
            }, f, indent=2)
    
    @classmethod
    def load(cls, str path):
        """Load tokenizer vocabulary from disk"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.special_tokens = data["special_tokens"]
        
        # Rebuild id_to_token mapping
        tokenizer.id_to_token = {int(idx): token for token, idx in tokenizer.token_to_id.items()}
        
        return tokenizer
