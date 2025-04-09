import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

from transformer import TransformerBlock, TransformerEncoder, TransformerDecoder, Transformer


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


class TestTransformer(unittest.TestCase):
    def setUp(self):
        # Common test parameters
        self.batch_size = 2
        self.seq_len = 10
        self.cond_len = 5
        self.d_model = 32
        self.vocab_size = 1000
        
        # Create a transformer with small dimensions for testing
        self.transformer = Transformer(
            d_model=self.d_model,
            depth_enc=2,
            depth_dec=2,
            heads=4,
            vocab=self.vocab_size,
        )
        
        # Create sample inputs
        self.x_tkns = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.z_tkns = torch.randint(0, self.vocab_size, (self.batch_size, self.cond_len))
    
    def test_initialization(self):
        """Test that the transformer is initialized correctly."""
        self.assertEqual(self.transformer.d_model, self.d_model)
        self.assertIsInstance(self.transformer.encoder, TransformerEncoder)
        self.assertIsInstance(self.transformer.decoder, TransformerDecoder)
    
    def test_forward_pass(self):
        """Test the full forward pass through the transformer."""
        output = self.transformer(self.x_tkns, self.z_tkns)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.vocab_size))
        
        # Test that outputs are different for different inputs
        output2 = self.transformer(
            torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len)), 
            self.z_tkns)
        self.assertFalse(torch.allclose(output, output2))
    
    def test_conditioning_effect(self):
        """Test that different conditioning tensors produce different outputs."""
        # Get output with original conditioning
        output1 = self.transformer(self.x_tkns, self.z_tkns)
        
        # Get output with different conditioning
        different_z = torch.randint(0, self.vocab_size, (self.batch_size, self.cond_len))
        output2 = self.transformer(self.x_tkns, different_z)
        
        # Check that outputs are different (conditioning has an effect)
        self.assertFalse(torch.allclose(output1, output2))
    
    def test_batch_processing(self):
        """Test that the transformer can process multiple sequences in a batch."""
        # Process a single item
        single_output = self.transformer(
            self.x_tkns[0:1], 
            self.z_tkns[0:1]
        )
        
        # Check shape
        self.assertEqual(single_output.shape, (1, self.seq_len, self.vocab_size))
        
        # Process the whole batch
        batch_output = self.transformer(self.x_tkns, self.z_tkns)
        
        # Check that the first item in the batch matches when processed individually
        # (allowing for minor numerical differences)
        self.assertTrue(torch.allclose(single_output[0], batch_output[0], atol=1e-6))
    
    def test_training_mode(self):
        """Test that the model works in training mode and parameters are updated."""
        # Get initial parameters
        initial_params = torch.nn.utils.parameters_to_vector(self.transformer.parameters()).detach().clone()
        
        # Set to training mode
        self.transformer.train()
        
        # Forward pass and backward
        output = self.transformer(self.x_tkns, self.z_tkns)
        loss = output.mean()
        loss.backward()
        
        # Apply a simple optimizer step
        with torch.no_grad():
            for param in self.transformer.parameters():
                if param.grad is not None:
                    param.data.add_(param.grad, alpha=-0.01)
        
        # Check that parameters changed
        final_params = torch.nn.utils.parameters_to_vector(self.transformer.parameters())
        self.assertFalse(torch.allclose(initial_params, final_params))

    def test_overfitting(self):
        print('Testing overfitting')
        FAKE_X = torch.randint(0, self.vocab_size, (10, self.seq_len))
        FAKE_Z = torch.randint(0, self.vocab_size, (10, self.cond_len))
        model = Transformer(
            d_model=64,
            depth_enc=4,
            depth_dec=4,
            heads=4,
            vocab=self.vocab_size,
        )

        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        for i in range(500):
            inp, trg = FAKE_X[:, :-1], FAKE_X[:, 1:]
            pred = model(inp, FAKE_Z)

            loss = F.cross_entropy(pred.flatten(0, 1), trg.flatten(0, 1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            acc = (pred.argmax(dim=-1)==trg).float().mean() * 100.
            if i % 20 == 0:
                print(f'[{i}/500] Acc={acc.item():.1f}% Loss={loss.item():.2f}')
        
        self.assertEqual(acc, 100)


if __name__ == '__main__':
    unittest.main()