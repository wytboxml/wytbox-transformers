import torch
import unittest
from torch import nn
from feedforward import FeedForward
from torch.testing import assert_close


class TestFeedForward(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Common test parameters
        self.batch_size = 8
        self.seq_len = 10
        self.dim = 64
        
    def test_initialization(self):
        """Test that the module initializes correctly with different parameters."""
        # Test default parameters
        ff = FeedForward(dim=self.dim)
        self.assertEqual(ff.fc1.in_features, self.dim)
        self.assertEqual(ff.fc1.out_features, self.dim * 4)
        self.assertEqual(ff.fc2.in_features, self.dim * 4)
        self.assertEqual(ff.fc2.out_features, self.dim)
        self.assertEqual(ff.dropout.p, 0.0)
        
        # Test custom parameters
        custom_expansion = 2
        custom_dropout = 0.5
        ff_custom = FeedForward(dim=self.dim, expansion=custom_expansion, drop=custom_dropout)
        self.assertEqual(ff_custom.fc1.out_features, self.dim * custom_expansion)
        self.assertEqual(ff_custom.dropout.p, custom_dropout)
    
    def test_forward_shape(self):
        """Test that the output shape matches the input shape."""
        ff = FeedForward(dim=self.dim)
        
        # Test with 3D input (batch, seq, dim)
        x_3d = torch.randn(self.batch_size, self.seq_len, self.dim)
        output_3d = ff(x_3d)
        self.assertEqual(output_3d.shape, x_3d.shape)
        
        # Test with 2D input (batch, dim)
        x_2d = torch.randn(self.batch_size, self.dim)
        output_2d = ff(x_2d)
        self.assertEqual(output_2d.shape, x_2d.shape)
    
    def test_forward_computation(self):
        """Test the computation logic of the forward pass with a controlled input."""
        # Create a feed-forward network with no dropout for deterministic testing
        ff = FeedForward(dim=4, expansion=2, drop=0.0)
        
        # Set weights to known values for predictable output
        with torch.no_grad():
            # Set fc1 weights to identity-like matrix with some values
            ff.fc1.weight.fill_(0.0)
            ff.fc1.bias.fill_(0.1)
            torch.diagonal(ff.fc1.weight)[:] = 1.0
            
            # Set fc2 weights to identity-like matrix
            ff.fc2.weight.fill_(0.0)
            ff.fc2.bias.fill_(0.0)
            torch.diagonal(ff.fc2.weight)[:] = 0.5
        
        # Create a simple input tensor
        x = torch.ones(2, 4)
        
        # Expected computation:
        # 1. fc1: x * 1.0 + 0.1 = 1.1
        # 2. ReLU: max(0, 1.1) = 1.1
        # 3. fc2: 1.1 * 0.5 + 0.0 = 0.55
        expected = torch.full((2, 4), 0.55)
        
        # Get actual output
        output = ff(x)
        
        # Compare outputs
        assert_close(output, expected)
    
    def test_dropout(self):
        """Test that dropout behaves differently in training vs evaluation modes."""
        # High dropout for clearer effect
        ff = FeedForward(dim=self.dim, drop=0.9)
        
        # Same input for both tests
        x = torch.ones(self.batch_size, self.dim)
        
        # In training mode, outputs should differ due to dropout
        ff.train()
        output1 = ff(x)
        output2 = ff(x)
        # Outputs should be different in training mode
        self.assertFalse(torch.allclose(output1, output2))
        
        # In evaluation mode, outputs should be deterministic (no dropout)
        ff.eval()
        output1 = ff(x)
        output2 = ff(x)
        # Outputs should be identical in eval mode
        self.assertTrue(torch.allclose(output1, output2))

if __name__ == "__main__":
    unittest.main()