import unittest
import torch
import torch.nn as nn
from torch.testing import assert_close

from transformer import TransformerBlock, TransformerEncoder, TransformerDecoder


class TestTransformerBlock(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.heads = 4
        self.d_cond = 32
        self.batch_size = 2
        self.seq_len = 10
        self.cond_len = 5
        
        # Create transformer blocks for testing
        self.block_no_cond = TransformerBlock(
            self.d_model, self.heads
        )
        self.block_with_cond = TransformerBlock(
            self.d_model, self.heads, d_cond=self.d_cond,
        )
        
        # Test tensors
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.z = torch.randn(self.batch_size, self.cond_len, self.d_cond)
    
    def test_forward_no_cond(self):
        """Test forward pass without conditioning"""
        output = self.block_no_cond(self.x)
        
        # Check output shape
        self.assertEqual(output.shape, self.x.shape)
        
        # Ensure output is not just the input (transformation happened)
        self.assertFalse(torch.allclose(output, self.x))
    
    def test_forward_with_cond(self):
        """Test forward pass with conditioning"""
        output = self.block_with_cond(self.x, self.z)
        
        # Check output shape
        self.assertEqual(output.shape, self.x.shape)
        
        # Ensure output is not just the input (transformation happened)
        self.assertFalse(torch.allclose(output, self.x))
    
    def test_forward_missing_cond(self):
        """Test error when conditioning is expected but not provided"""
        with self.assertRaises(AssertionError):
            self.block_with_cond(self.x)
    
    def test_forward_unexpected_cond(self):
        """Test that providing conditioning when not expected is handled gracefully"""
        with self.assertRaises(AssertionError):
            self.block_no_cond(self.x, self.z)


class TestTransformerEncoder(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.depth = 3
        self.heads = 4
        self.vocab = 1000
        self.d_out = 32
        self.batch_size = 2
        self.seq_len = 10
        
        # Create encoders for testing
        self.encoder_no_proj = TransformerEncoder(
            self.d_model, self.depth, self.heads, self.vocab, 
            d_out=None, drop_attn=0.1, drop_ffn=0.1
        )
        self.encoder_with_proj = TransformerEncoder(
            self.d_model, self.depth, self.heads, self.vocab, 
            d_out=self.d_out, drop_attn=0.1, drop_ffn=0.1
        )
        
        # Test tensors - token indices
        self.x_tkns = torch.randint(0, self.vocab, (self.batch_size, self.seq_len))
    
    def test_init(self):
        """Test that the encoder initializes correctly with and without projection"""
        # Test common components
        self.assertIsInstance(self.encoder_no_proj.tkn_embed, nn.Embedding)
        self.assertEqual(self.encoder_no_proj.tkn_embed.num_embeddings, self.vocab)
        self.assertEqual(self.encoder_no_proj.tkn_embed.embedding_dim, self.d_model)
        
        self.assertEqual(len(self.encoder_no_proj.blocks), self.depth)
        
        # Test without projection
        self.assertIsNone(self.encoder_no_proj.norm)
        self.assertIsNone(self.encoder_no_proj.fc)
        
        # Test with projection
        self.assertIsInstance(self.encoder_with_proj.norm, nn.LayerNorm)
        self.assertIsInstance(self.encoder_with_proj.fc, nn.Linear)
        self.assertEqual(self.encoder_with_proj.fc.in_features, self.d_model)
        self.assertEqual(self.encoder_with_proj.fc.out_features, self.d_out)
    
    def test_forward_no_proj(self):
        """Test forward pass without projection"""
        output = self.encoder_no_proj(self.x_tkns)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)
    
    def test_forward_with_proj(self):
        """Test forward pass with projection"""
        output = self.encoder_with_proj(self.x_tkns)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.d_out)
        self.assertEqual(output.shape, expected_shape)


class TestTransformerDecoder(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.d_cond = 32
        self.depth = 3
        self.heads = 4
        self.vocab = 1000
        self.d_out = 32
        self.batch_size = 2
        self.seq_len = 10
        self.cond_len = 5
        
        # Create decoders for testing
        self.decoder_no_proj = TransformerDecoder(
            self.d_model, self.d_cond, self.depth, self.heads, self.vocab, 
            d_out=None, drop_attn=0.1, drop_xattn=0.2, drop_ffn=0.1
        )
        self.decoder_with_proj = TransformerDecoder(
            self.d_model, self.d_cond, self.depth, self.heads, self.vocab, 
            d_out=self.d_out, drop_attn=0.1, drop_xattn=0.2, drop_ffn=0.1
        )
        
        # Test tensors
        self.x_tkns = torch.randint(0, self.vocab, (self.batch_size, self.seq_len))
        self.z = torch.randn(self.batch_size, self.cond_len, self.d_cond)
    
    def test_init(self):
        """Test that the decoder initializes correctly with and without projection"""
        # Test common components
        self.assertIsInstance(self.decoder_no_proj.tkn_embed, nn.Embedding)
        self.assertEqual(self.decoder_no_proj.tkn_embed.num_embeddings, self.vocab)
        self.assertEqual(self.decoder_no_proj.tkn_embed.embedding_dim, self.d_model)
        
        self.assertEqual(len(self.decoder_no_proj.blocks), self.depth)
        
        # Test without projection
        self.assertIsNone(self.decoder_no_proj.norm)
        self.assertIsNone(self.decoder_no_proj.fc)
        
        # Test with projection
        self.assertIsInstance(self.decoder_with_proj.norm, nn.LayerNorm)
        self.assertIsInstance(self.decoder_with_proj.fc, nn.Linear)
        self.assertEqual(self.decoder_with_proj.fc.in_features, self.d_model)
        self.assertEqual(self.decoder_with_proj.fc.out_features, self.d_out)
    
    def test_forward_no_proj(self):
        """Test forward pass without projection"""
        output = self.decoder_no_proj(self.x_tkns, self.z)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)
    
    def test_forward_with_proj(self):
        """Test forward pass with projection"""
        output = self.decoder_with_proj(self.x_tkns, self.z)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.d_out)
        self.assertEqual(output.shape, expected_shape)
    
    def test_forward_missing_cond(self):
        """Test error when conditioning is not provided"""
        with self.assertRaises(TypeError):
            self.decoder_no_proj(self.x_tkns)

if __name__ == '__main__':
    unittest.main()