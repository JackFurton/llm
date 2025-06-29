import sys
import os
import unittest
import torch

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.transformer import (
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    PositionalEncoding,
    CustomLanguageModel
)

class TestTransformerComponents(unittest.TestCase):
    def test_multi_head_attention(self):
        batch_size = 2
        seq_len = 10
        d_model = 64
        num_heads = 8
        
        # Create attention module
        attention = MultiHeadAttention(d_model, num_heads)
        
        # Create input tensors
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output = attention(x, x, x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
    
    def test_feed_forward(self):
        batch_size = 2
        seq_len = 10
        d_model = 64
        d_ff = 256
        
        # Create feed-forward module
        ff = FeedForward(d_model, d_ff)
        
        # Create input tensor
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output = ff(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
    
    def test_transformer_block(self):
        batch_size = 2
        seq_len = 10
        d_model = 64
        num_heads = 8
        d_ff = 256
        
        # Create transformer block
        block = TransformerBlock(d_model, num_heads, d_ff)
        
        # Create input tensor
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output = block(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
    
    def test_positional_encoding(self):
        batch_size = 2
        seq_len = 10
        d_model = 64
        
        # Create positional encoding module
        pe = PositionalEncoding(d_model)
        
        # Create input tensor
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output = pe(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
        
        # Check that positional encoding was added
        self.assertFalse(torch.allclose(x, output))

class TestLanguageModel(unittest.TestCase):
    def test_model_forward(self):
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        d_model = 64
        num_heads = 8
        num_layers = 2
        d_ff = 256
        
        # Create model
        model = CustomLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff
        )
        
        # Create input tensor
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        output = model(input_ids)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, vocab_size))
    
    def test_model_generate(self):
        batch_size = 1
        seq_len = 5
        vocab_size = 1000
        d_model = 64
        num_heads = 8
        num_layers = 2
        d_ff = 256
        max_length = 10
        
        # Create model
        model = CustomLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff
        )
        
        # Create input tensor
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Generate
        output_ids = model.generate(input_ids, max_length=max_length)
        
        # Check output shape
        self.assertEqual(output_ids.shape, (batch_size, max_length))
        
        # Check that the first seq_len tokens are the same as input
        self.assertTrue(torch.all(output_ids[:, :seq_len] == input_ids))

if __name__ == '__main__':
    unittest.main()
