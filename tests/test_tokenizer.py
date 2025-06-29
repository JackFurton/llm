import sys
import os
import unittest
import tempfile

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.tokenizer import SimpleTokenizer, CharacterTokenizer

class TestSimpleTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SimpleTokenizer(vocab_size=100)
        self.sample_texts = [
            "This is a test sentence.",
            "Another example for tokenization.",
            "The quick brown fox jumps over the lazy dog."
        ]
        self.tokenizer.train(self.sample_texts)
    
    def test_encode_decode(self):
        text = "This is a test."
        encoded = self.tokenizer.encode(text)
        
        # Check that encoded is a list of integers
        self.assertTrue(all(isinstance(token_id, int) for token_id in encoded))
        
        # Decode back
        decoded = self.tokenizer.decode(encoded)
        
        # Check that the decoded text contains the original words
        # (might not be exact due to tokenization differences)
        for word in text.lower().split():
            self.assertIn(word, decoded)
    
    def test_special_tokens(self):
        # Check that special tokens are in the vocabulary
        for token in ["<pad>", "<unk>", "<bos>", "<eos>"]:
            self.assertIn(token, self.tokenizer.token_to_id)
    
    def test_save_load(self):
        # Save tokenizer
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            self.tokenizer.save(tmp.name)
            
            # Load tokenizer
            loaded_tokenizer = SimpleTokenizer.load(tmp.name)
            
            # Check that vocabularies match
            self.assertEqual(self.tokenizer.token_to_id, loaded_tokenizer.token_to_id)
            self.assertEqual(self.tokenizer.special_tokens, loaded_tokenizer.special_tokens)
    
    def test_unknown_tokens(self):
        # Test handling of unknown tokens
        text = "This is a completely unknown word: xyzabc123"
        encoded = self.tokenizer.encode(text)
        
        # Check that <unk> token is used
        self.assertIn(self.tokenizer.special_tokens["<unk>"], encoded)
        
        # Decode back
        decoded = self.tokenizer.decode(encoded)
        
        # Check that known words are preserved
        for word in ["this", "is", "a"]:
            self.assertIn(word, decoded)

class TestCharacterTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.sample_texts = [
            "Hello, world!",
            "Testing character tokenization.",
            "1234567890!@#$%^&*()"
        ]
        self.tokenizer.train(self.sample_texts)
    
    def test_encode_decode(self):
        text = "Hello, world!"
        encoded = self.tokenizer.encode(text)
        
        # Check that encoded is a list of integers
        self.assertTrue(all(isinstance(token_id, int) for token_id in encoded))
        
        # Decode back
        decoded = self.tokenizer.decode(encoded)
        
        # For character tokenizer, decoded should match original exactly
        # (except for special tokens)
        self.assertEqual(text, decoded)
    
    def test_special_tokens(self):
        # Check that special tokens are in the vocabulary
        for token in ["<pad>", "<unk>", "<bos>", "<eos>"]:
            self.assertIn(token, self.tokenizer.token_to_id)
    
    def test_save_load(self):
        # Save tokenizer
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            self.tokenizer.save(tmp.name)
            
            # Load tokenizer
            loaded_tokenizer = CharacterTokenizer.load(tmp.name)
            
            # Check that vocabularies match
            self.assertEqual(self.tokenizer.token_to_id, loaded_tokenizer.token_to_id)
            self.assertEqual(self.tokenizer.special_tokens, loaded_tokenizer.special_tokens)
    
    def test_all_characters_tokenized(self):
        # Test that all characters in the sample texts are tokenized
        for text in self.sample_texts:
            for char in text:
                self.assertIn(char, self.tokenizer.token_to_id)

if __name__ == '__main__':
    unittest.main()
