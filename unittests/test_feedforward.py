import torch
import unittest
from torch import nn
from feedforward import FeedForward

class TestFeedForward(unittest.TestCase):
    """
    Unit tests for the FeedForward module.
    """
    
    def setUp(self):
        # Set up common test parameters
        self.dim = 64
        self.batch_size = 8
        self.seq_len = 10
        self.expansion = 4
        self.drop = 0.0  # Set to 0 for deterministic testing
        
        # Create module instance for testing
        self.ff = FeedForward(dim=self.dim, expansion=self.expansion, drop=self.drop)
        
        # Create sample input
        self.input = torch.randn(self.batch_size, self.seq_len, self.dim)
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        output = self.ff(self.input)
        self.assertEqual(output.shape, self.input.shape)
    
    
if __name__ == "__main__":
    unittest.main()