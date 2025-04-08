import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import math

import sys
from position_embeds import SinusoidalPEs


class TestSinusoidalPEs(unittest.TestCase):
    def setUp(self):
        # Common dimensions used in tests
        self.dim = 512
        self.pos_encoder = SinusoidalPEs(dim=self.dim)
    
    def test_initialization(self):
        """Test if the omega buffer is initialized correctly."""
        # Check omega shape
        omega = self.pos_encoder.omega.squeeze(0)
        self.assertEqual(omega.shape, (self.dim // 2,))
        
        # Check omega values (decreasing exponentially)
        # First value should be close to 1.0
        self.assertAlmostEqual(omega[0].item(), 1.0, places=5)
        # Last value should be close to 1/10000
        self.assertAlmostEqual(omega[-1].item(), 1/10000, places=5)
        
        # Check that frequencies decrease exponentially
        log_ratios = torch.log(omega[:-1] / omega[1:])
        self.assertTrue(torch.allclose(log_ratios, log_ratios[0], rtol=1e-3))
    
    def test_forward_with_length(self):
        """Test forward pass using sequence length L."""
        L = 10
        pos_embeds = self.pos_encoder(L=L)
        
        # Check output shape
        self.assertEqual(pos_embeds.shape, (L, self.dim))
        
        # Manual computation for verification
        positions = torch.arange(L).unsqueeze(-1)  # L, 1
        omega = self.pos_encoder.omega  # 1, dim//2
        expected_sin = torch.sin(positions * omega)
        expected_cos = torch.cos(positions * omega)
        expected = torch.cat((expected_sin, expected_cos), dim=-1)
        
        # Check if output matches manual computation
        self.assertTrue(torch.allclose(pos_embeds, expected))
    
    def test_forward_with_positions(self):
        """Test forward pass using explicit positions."""
        # Create positions tensor with batch dimension
        pos = torch.tensor([0, 5, 10])  # 3
        pos_embeds = self.pos_encoder(pos=pos)
        
        # Check output shape (should preserve batch dimension)
        self.assertEqual(pos_embeds.shape, (3, self.dim))
        
        # Check specific positions manually
        manual_embeds = []
        for p in pos.squeeze(-1):
            p_tensor = torch.tensor([p.item()]).unsqueeze(-1)  # 1, 1
            sin_part = torch.sin(p_tensor * self.pos_encoder.omega)
            cos_part = torch.cos(p_tensor * self.pos_encoder.omega)
            combined = torch.cat((sin_part, cos_part), dim=-1)
            manual_embeds.append(combined)
        expected = torch.cat(manual_embeds, dim=0)
        self.assertTrue(torch.allclose(pos_embeds, expected))

        # Check that outputs using positions or lenght match
        expected = self.pos_encoder(L=11)[pos]
        self.assertTrue(torch.allclose(pos_embeds, expected))
    
    def test_multi_dimensional_positions(self):
        """Test with multi-dimensional position tensors."""
        # Create 2D positions tensor (batch, seq_len)
        pos = torch.tensor([[0, 1, 2], [3, 4, 5]])  # 2, 3
        pos_embeds = self.pos_encoder(pos=pos)
        
        # Check output shape (should be batch, seq_len, dim)
        self.assertEqual(pos_embeds.shape, (2, 3, self.dim))
        
        # Verify first and last batch embeddings
        pos_0 = pos[0].unsqueeze(-1)  # First batch, shape: 3, 1
        expected_0 = torch.cat((
            torch.sin(pos_0 * self.pos_encoder.omega),
            torch.cos(pos_0 * self.pos_encoder.omega)
        ), dim=-1)
        
        self.assertTrue(torch.allclose(pos_embeds[0], expected_0))
            
    def test_input_validation(self):
        """Test input validation."""
        # Should raise an assertion error when both L and pos are None
        with self.assertRaises(AssertionError):
            self.pos_encoder(L=None, pos=None)

        # Create encoder with odd dimension
        with self.assertRaises(AssertionError):
            SinusoidalPEs(dim=513)
    
if __name__ == '__main__':
    unittest.main()