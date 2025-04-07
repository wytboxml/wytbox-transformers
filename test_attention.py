import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

from attention import _check_attention_dims, SoftmaxAttention, MHSA, MHCA


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


class TestMultiHeadAttention(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.dim = 64
        self.heads = 4
        
        # Fix random seed for reproducibility
        torch.manual_seed(42)
        
        # Create test inputs
        self.x = torch.randn(self.batch_size, self.seq_len, self.dim)
        self.z = torch.randn(self.batch_size, self.seq_len * 2, self.dim)  # Different sequence length
        
    def test_mhsa_forward_shape(self):
        """Test MHSA forward pass output shapes."""
        mhsa = MHSA(dim=self.dim, heads=self.heads)
        output = mhsa(self.x)
        
        # Output should have same shape as input
        self.assertEqual(output.shape, self.x.shape)
        
        # Test with return_attn=True
        output, attn = mhsa(self.x, return_attn=True)
        self.assertEqual(output.shape, self.x.shape)
        self.assertEqual(attn.shape, (self.batch_size, self.heads, self.seq_len, self.seq_len))
        
    def test_mhxa_forward_shape(self):
        """Test MHXA forward pass output shapes."""
        mhca = MHCA(dim=self.dim, heads=self.heads)
        output = mhca(self.x, self.z)
        
        # Output should have same shape as query input except for feature dimension
        self.assertEqual(output.shape, self.x.shape)
        
        # Test with return_attn=True
        output, attn = mhca(self.x, self.z, return_attn=True)
        self.assertEqual(output.shape, self.x.shape)
        self.assertEqual(attn.shape, (self.batch_size, self.heads, self.seq_len, self.seq_len * 2))
        
    def test_mhsa_equivalence(self):
        """Test that MHSA is equivalent to MHXA when using same input for query and key/value."""
        torch.manual_seed(42)  # Ensure deterministic initialization
        mhsa = MHSA(dim=self.dim, heads=self.heads)
        
        torch.manual_seed(42)  # Same initialization for fair comparison
        mhca = MHCA(dim=self.dim, heads=self.heads)
        
        # Replace MHCA's projections to match MHSA's combined projection
        # This is a hack for testing equivalence - in practice the weights would be trained differently
        with torch.no_grad():
            # Extract weights and biases from MHSA's qkv projection
            qkv_weight = mhsa.qkv_projection.weight
            qkv_bias = mhsa.qkv_projection.bias
            
            # Match projections in MHCA and MHSA
            mhca.q_projection.weight.copy_(qkv_weight[:self.dim])
            mhca.q_projection.bias.copy_(qkv_bias[:self.dim])
            mhca.kv_projection.weight.copy_(qkv_weight[self.dim:])
            mhca.kv_projection.bias.copy_(qkv_bias[self.dim:])
            mhca.out_projection.weight.copy_(mhsa.out_projection.weight)
            mhca.out_projection.bias.copy_(mhsa.out_projection.bias)
        
        # With identical weights, outputs should be the same when input is the same
        with torch.no_grad():
            output_mhsa = mhsa(self.x)
            output_mhca = mhca(self.x, self.x)
        
        self.assertTrue(torch.allclose(output_mhsa, output_mhca, atol=1e-5))
        
    def test_batched(self):
        """Test MHSA and MHCA with additional batch dimensions."""
        # Create input with extra batch dimension
        # Shape: (extra_batch, batch_size, seq_len, dim)
        extra_batch = 3
        x_batched = torch.randn(extra_batch, self.batch_size, self.seq_len, self.dim)
        z_batched = torch.randn(extra_batch, self.batch_size, 12, self.dim)
        
        mhsa = MHSA(dim=self.dim, heads=self.heads)
        output_mhsa = mhsa(x_batched)
        
        mhca = MHCA(dim=self.dim, heads=self.heads)
        output_mhca = mhca(x_batched, z_batched)
        
        # Output should preserve all input dimensions
        self.assertEqual(output_mhsa.shape, x_batched.shape)
        self.assertEqual(output_mhca.shape, x_batched.shape)
        
        
if __name__ == '__main__':
    unittest.main()