import re
import logging
import string
from typing import List, Dict, Any, Optional, Set, Union
from collections import Counter
import hashlib

from .preprocessor import TextProcessor

# Configure logging
logger = logging.getLogger('text_preprocessor.filters')

class LanguageFilter(TextProcessor):
    """
    Filter text based on language detection
    """
    
    def __init__(self, allowed_languages: List[str] = ["en"]):
        super().__init__(name="language_filter")
        self.allowed_languages = set(allowed_languages)
        
        # Try to import language detection libraries
        try:
            import langdetect
            self.langdetect = langdetect
            self._detect_func = self._detect_with_langdetect
            logger.info("Using langdetect for language detection")
        except ImportError:
            logger.warning("langdetect not available, falling back to simple heuristics")
            self._detect_func = self._detect_with_heuristics
    
    def _detect_with_langdetect(self, text: str) -> str:
        """Detect language using langdetect library"""
        try:
            # Use only the first 10000 characters for faster detection
            return self.langdetect.detect(text[:10000])
        except:
            return "unknown"
    
    def _detect_with_heuristics(self, text: str) -> str:
        """
        Simple language detection heuristic based on character frequencies
        Only distinguishes between a few languages
        """
        # English letter frequencies
        en_freqs = {
            'e': 12.7, 't': 9.1, 'a': 8.2, 'o': 7.5, 'i': 7.0, 
            'n': 6.7, 's': 6.3, 'h': 6.1, 'r': 6.0, 'd': 4.3
        }
        
        # Spanish letter frequencies
        es_freqs = {
            'e': 13.7, 'a': 12.5, 'o': 8.7, 's': 7.9, 'r': 6.9,
            'n': 6.7, 'i': 6.3, 'd': 5.9, 'l': 4.9, 'c': 4.7
        }
        
        # French letter frequencies
        fr_freqs = {
            'e': 14.7, 'a': 7.6, 's': 7.9, 'i': 7.5, 't': 7.2,
            'n': 7.1, 'r': 6.6, 'u': 6.3, 'l': 5.8, 'o': 5.4
        }
        
        # German letter frequencies
        de_freqs = {
            'e': 16.4, 'n': 9.8, 'i': 7.6, 's': 7.3, 'r': 7.0,
            't': 6.1, 'a': 6.5, 'd': 5.1, 'h': 4.8, 'u': 4.4
        }
        
        # Count letter frequencies in the text
        text = text.lower()
        total_chars = sum(1 for c in text if c.isalpha())
        if total_chars == 0:
            return "unknown"
        
        counter = Counter(c for c in text if c.isalpha())
        text_freqs = {c: (count / total_chars) * 100 for c, count in counter.items()}
        
        # Calculate distance to each language
        en_dist = sum((text_freqs.get(c, 0) - freq) ** 2 for c, freq in en_freqs.items())
        es_dist = sum((text_freqs.get(c, 0) - freq) ** 2 for c, freq in es_freqs.items())
        fr_dist = sum((text_freqs.get(c, 0) - freq) ** 2 for c, freq in fr_freqs.items())
        de_dist = sum((text_freqs.get(c, 0) - freq) ** 2 for c, freq in de_freqs.items())
        
        # Find the language with the smallest distance
        min_dist = min(en_dist, es_dist, fr_dist, de_dist)
        
        if min_dist == en_dist:
            return "en"
        elif min_dist == es_dist:
            return "es"
        elif min_dist == fr_dist:
            return "fr"
        elif min_dist == de_dist:
            return "de"
        else:
            return "unknown"
    
    def process(self, text: str) -> bool:
        """
        Check if the text is in an allowed language
        
        Args:
            text: Input text
            
        Returns:
            True if the text is in an allowed language, False otherwise
        """
        # Skip very short texts
        if len(text) < 100:
            return True
        
        # Detect language
        lang = self._detect_func(text)
        
        # Check if language is allowed
        is_allowed = lang in self.allowed_languages
        
        if not is_allowed:
            logger.info(f"Filtered text in language: {lang}")
        
        return is_allowed
    
    def should_process(self, text: str) -> bool:
        """Only process texts with sufficient length"""
        return len(text) >= 100


class ContentFilter(TextProcessor):
    """
    Filter text based on content rules (profanity, sensitive topics, etc.)
    """
    
    def __init__(self, 
                 profanity_file: Optional[str] = None,
                 sensitive_topics: Optional[List[str]] = None,
                 max_profanity_ratio: float = 0.01):
        super().__init__(name="content_filter")
        
        # Load profanity list
        self.profanity_words = set()
        if profanity_file:
            try:
                with open(profanity_file, 'r', encoding='utf-8') as f:
                    self.profanity_words = set(line.strip().lower() for line in f if line.strip())
            except Exception as e:
                logger.error(f"Error loading profanity file: {e}")
        
        # Default profanity list if none provided or loading failed
        if not self.profanity_words:
            # Very minimal list for demonstration purposes
            self.profanity_words = {
                "profanity1", "profanity2", "profanity3", "profanity4", "profanity5"
            }
        
        # Sensitive topics
        self.sensitive_topics = sensitive_topics or [
            "politics", "religion", "violence", "terrorism", "suicide",
            "abuse", "discrimination", "racism", "sexism"
        ]
        
        # Compile regex patterns for sensitive topics
        self.sensitive_patterns = [
            re.compile(r'\b' + re.escape(topic) + r'\b', re.IGNORECASE)
            for topic in self.sensitive_topics
        ]
        
        # Maximum allowed profanity ratio
        self.max_profanity_ratio = max_profanity_ratio
    
    def process(self, text: str) -> bool:
        """
        Check if the text passes content filtering
        
        Args:
            text: Input text
            
        Returns:
            True if the text passes filtering, False otherwise
        """
        # Check profanity
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return True
        
        profanity_count = sum(1 for word in words if word in self.profanity_words)
        profanity_ratio = profanity_count / len(words)
        
        if profanity_ratio > self.max_profanity_ratio:
            logger.info(f"Filtered text with profanity ratio: {profanity_ratio:.4f}")
            return False
        
        # Check sensitive topics
        sensitive_matches = []
        for i, pattern in enumerate(self.sensitive_patterns):
            if pattern.search(text):
                sensitive_matches.append(self.sensitive_topics[i])
        
        # If too many sensitive topics are found, filter the text
        if len(sensitive_matches) > 3:
            logger.info(f"Filtered text with sensitive topics: {', '.join(sensitive_matches)}")
            return False
        
        return True


class QualityFilter(TextProcessor):
    """
    Filter text based on quality metrics
    """
    
    def __init__(self, 
                 min_length: int = 100,
                 max_length: int = 100000,
                 min_avg_word_length: float = 3.0,
                 max_avg_word_length: float = 10.0,
                 min_sentence_count: int = 3,
                 max_repetition_ratio: float = 0.3):
        super().__init__(name="quality_filter")
        
        self.min_length = min_length
        self.max_length = max_length
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length
        self.min_sentence_count = min_sentence_count
        self.max_repetition_ratio = max_repetition_ratio
    
    def process(self, text: str) -> bool:
        """
        Check if the text meets quality standards
        
        Args:
            text: Input text
            
        Returns:
            True if the text passes quality checks, False otherwise
        """
        # Check text length
        if len(text) < self.min_length:
            logger.info(f"Filtered text that is too short: {len(text)} chars")
            return False
        
        if len(text) > self.max_length:
            logger.info(f"Filtered text that is too long: {len(text)} chars")
            return False
        
        # Check average word length
        words = re.findall(r'\b\w+\b', text)
        if not words:
            logger.info("Filtered text with no words")
            return False
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length < self.min_avg_word_length:
            logger.info(f"Filtered text with low avg word length: {avg_word_length:.2f}")
            return False
        
        if avg_word_length > self.max_avg_word_length:
            logger.info(f"Filtered text with high avg word length: {avg_word_length:.2f}")
            return False
        
        # Check sentence count
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < self.min_sentence_count:
            logger.info(f"Filtered text with too few sentences: {len(sentences)}")
            return False
        
        # Check for repetition
        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1] if word_counts else 0
        repetition_ratio = most_common_count / len(words) if words else 1.0
        
        if repetition_ratio > self.max_repetition_ratio:
            logger.info(f"Filtered text with high repetition ratio: {repetition_ratio:.2f}")
            return False
        
        return True


class DuplicateFilter(TextProcessor):
    """
    Filter duplicate or near-duplicate text
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        super().__init__(name="duplicate_filter")
        self.similarity_threshold = similarity_threshold
        self.document_hashes = set()
        self.minhash_signatures = {}
    
    def _compute_hash(self, text: str) -> str:
        """Compute a hash of the text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _compute_shingles(self, text: str, k: int = 5) -> Set[str]:
        """Compute k-shingles (character n-grams) from text"""
        return {text[i:i+k] for i in range(len(text) - k + 1)}
    
    def _compute_minhash(self, shingles: Set[str], num_hashes: int = 100) -> List[int]:
        """Compute MinHash signature for a set of shingles"""
        import random
        
        # Use deterministic hash functions
        hash_funcs = [
            lambda x, i: hash(x + str(i)) % 10000
            for i in range(num_hashes)
        ]
        
        # Compute signature
        signature = []
        for h in hash_funcs:
            min_hash = float('inf')
            for shingle in shingles:
                hash_val = h(shingle, len(signature))
                min_hash = min(min_hash, hash_val)
            signature.append(min_hash)
        
        return signature
    
    def _jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures"""
        if not sig1 or not sig2:
            return 0.0
        
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    def process(self, text: str) -> bool:
        """
        Check if the text is a duplicate
        
        Args:
            text: Input text
            
        Returns:
            True if the text is not a duplicate, False otherwise
        """
        # Compute hash
        text_hash = self._compute_hash(text)
        
        # Check for exact duplicates
        if text_hash in self.document_hashes:
            logger.info("Filtered exact duplicate text")
            return False
        
        # For short texts, only check exact duplicates
        if len(text) < 1000:
            self.document_hashes.add(text_hash)
            return True
        
        # For longer texts, check for near-duplicates using MinHash
        try:
            # Compute shingles and MinHash signature
            shingles = self._compute_shingles(text)
            signature = self._compute_minhash(shingles)
            
            # Check similarity with existing documents
            for doc_id, existing_sig in self.minhash_signatures.items():
                similarity = self._jaccard_similarity(signature, existing_sig)
                if similarity > self.similarity_threshold:
                    logger.info(f"Filtered near-duplicate text (similarity: {similarity:.2f})")
                    return False
            
            # Add to known documents
            self.document_hashes.add(text_hash)
            self.minhash_signatures[text_hash] = signature
            
            return True
            
        except Exception as e:
            logger.error(f"Error in duplicate detection: {e}")
            # Fall back to exact duplicate detection
            self.document_hashes.add(text_hash)
            return True
