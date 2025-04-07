import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

from attention import _check_attention_dims, SoftmaxAttention


class TestAttentionMechanism(unittest.TestCase):
    def test_check_attention_dims_valid(self):
        """Test that valid dimensions pass the check without errors."""
        # Single batch, single head case
        q = torch.randn(10, 64)  # Lq=10, dim=64
        k = torch.randn(15, 64)  # Lk=15, dim=64
        v = torch.randn(15, 32)  # Lk=15, dim_v=32
        try:
            _check_attention_dims(q, k, v)
        except ValueError:
            self.fail("_check_attention_dims raised ValueError unexpectedly for valid dimensions")
        
        # Batch dimension case
        q = torch.randn(32, 10, 64)  # batch=32, Lq=10, dim=64
        k = torch.randn(32, 15, 64)  # batch=32, Lk=15, dim=64
        v = torch.randn(32, 15, 32)  # batch=32, Lk=15, dim_v=32
        try:
            _check_attention_dims(q, k, v)
        except ValueError:
            self.fail("_check_attention_dims raised ValueError unexpectedly for valid batch dimensions")
        
        # Multi-head case (heads as batch dimension)
        q = torch.randn(32, 8, 10, 64)  # batch=32, heads=8, Lq=10, dim=64
        k = torch.randn(32, 8, 15, 64)  # batch=32, heads=8, Lk=15, dim=64
        v = torch.randn(32, 8, 15, 32)  # batch=32, heads=8, Lk=15, dim_v=32
        try:
            _check_attention_dims(q, k, v)
        except ValueError:
            self.fail("_check_attention_dims raised ValueError unexpectedly for valid multi-head dimensions")

    def test_check_attention_invalid_ndims(self):
        """Test that error is raised when dimensions don't match."""
        q = torch.randn(10, 64)      # 2D tensor
        k = torch.randn(32, 15, 64)  # 3D tensor
        v = torch.randn(15, 32)      # 2D tensor
        
        with self.assertRaises(ValueError) as context:
            _check_attention_dims(q, k, v)
        self.assertTrue('same number of dimensions' in str(context.exception))

        # Batch dimensions don't match.
        q = torch.randn(32, 10, 64)  # batch=32
        k = torch.randn(16, 15, 64)  # batch=16 (different from q)
        v = torch.randn(16, 15, 32)  # batch=16
        
        with self.assertRaises(ValueError) as context:
            _check_attention_dims(q, k, v)
        self.assertTrue('Batch sizes' in str(context.exception))

        #  q and k don't have same feature dimension.
        q = torch.randn(32, 10, 64)  # dim=64
        k = torch.randn(32, 15, 48)  # dim=48 (different from q)
        v = torch.randn(32, 15, 32)  # dim_v=32
        
        with self.assertRaises(ValueError) as context:
            _check_attention_dims(q, k, v)
        self.assertTrue('same space' in str(context.exception))

        # keys and values have different sequence lengths.
        q = torch.randn(32, 10, 64)  # Lq=10
        k = torch.randn(32, 15, 64)  # Lk=15
        v = torch.randn(32, 12, 32)  # Lv=12 (different from k)
        
        with self.assertRaises(ValueError) as context:
            _check_attention_dims(q, k, v)
        self.assertTrue('same sequence length' in str(context.exception))

    def test_attention_computation(self):
        """Test that attention computation produces expected output shape without custom scale."""
        bs = 2
        Lq = 3
        Lk = 4
        dim = 8
        dim_v = 16
        
        q = torch.randn(bs, Lq, dim)
        k = torch.randn(bs, Lk, dim)
        v = torch.randn(bs, Lk, dim_v)
        
        attention = SoftmaxAttention()
        output, attn = attention(q, k, v, return_attn=True)
        
        # Check outputs
        self.assertEqual(output.shape, (bs, Lq, dim_v))
        self.assertEqual(attn.shape, (bs, Lq, Lk))
        self.assertTrue(torch.allclose(attn.sum(-1), torch.ones(bs, Lq), atol=1e-6))
        self.assertEqual(attention.scale, dim ** (-0.5))

        # Multi-head case
        num_heads = 4
        q = torch.randn(bs, num_heads, Lq, dim)
        k = torch.randn(bs, num_heads, Lk, dim)
        v = torch.randn(bs, num_heads, Lk, dim_v)
        
        # Check output shape includes the head dimension
        attention = SoftmaxAttention()
        output = attention(q, k, v)
        self.assertEqual(output.shape, (bs, num_heads, Lq, dim_v))

    def test_attention_matrix_values(self):
        """Test that attention values are computed correctly."""
        # Simple case with known values
        q = torch.tensor([[[1.0, 0.0]]])  # batch=1, seq_len=1, dim=2
        k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # batch=1, seq_len=2, dim=2
        v = torch.tensor([[[1.0], [2.0]]])  # batch=1, seq_len=2, dim_v=1
        
        attention = SoftmaxAttention(scale=1.0)  # Use scale=1.0 for simpler verification
        output, attn_weights = attention(q, k, v, return_attn=True)
        
        # Expected attention weights: softmax([1, 0]) = [e^1/(e^1+e^0), e^0/(e^1+e^0)] ≈ [0.731, 0.269]
        expected_attn = torch.tensor([[[0.7311, 0.2689]]])
        self.assertTrue(torch.allclose(attn_weights, expected_attn, atol=1e-4))
        
        # Expected output: [0.731*1.0 + 0.269*2.0] ≈ [1.269]
        expected_output = torch.tensor([[[1.2689]]])
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

        # Simple case where attention must be almost diagonal
        bs = 2
        L = 3
        dim = 8
        dim_v = 16
        q = F.normalize(torch.randn(bs, L, dim)) * 100
        k = q
        v = torch.randn(bs, L, dim_v)
        
        attention = SoftmaxAttention()
        output, attn_weights = attention(q, k, v, return_attn=True)
        
        # Expected attention weights: diagonal matrix
        expected_attn = torch.stack([torch.eye(L) for _ in range(bs)], dim=0)
        self.assertTrue(torch.allclose(attn_weights, expected_attn, atol=1e-4))
        
        # Expected output: Same as values
        self.assertTrue(torch.allclose(output, v, atol=1e-4))


if __name__ == '__main__':
    unittest.main()