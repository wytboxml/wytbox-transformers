import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *


def _check_attention_dims(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    Validates the dimensions of query, key, and value tensors for attention computation.
    
    This function checks that the attention tensors have compatible shapes according to
    the scaled dot-product attention requirements. It supports any number of batch 
    dimensions, including configurations where multiple attention heads are represented
    as additional batch dimensions.
    
    Args:
        q (torch.Tensor): Query tensor of shape (..., Lq, D)
        k (torch.Tensor): Key tensor of shape (..., Lk, D)
        v (torch.Tensor): Value tensor of shape (..., Lk, D_v)
        
    Raises:
        ValueError: If any of the following conditions are not met:
            - Q, K, V must have the same number of dimensions
            - All batch dimensions (all except the last two) must match
            - Q and K must have the same feature dimension (last dimension)
            - K and V must have the same sequence length (second-to-last dimension)
    """
    if not (q.ndim == k.ndim == v.ndim):
        raise ValueError('Q, K, V must have same number of dimensions.')
    
    for i in range(q.ndim-2):
        if not (q.shape[i] == k.shape[i] == v.shape[i]):
            raise ValueError('Batch sizes of Q, K, V must be the same.')
    
    if q.shape[-1] != k.shape[-1]:
        raise ValueError('Queries and Keys must be in the same space.')
    
    if k.shape[-2] != v.shape[-2]:
        raise ValueError('Keys and values must have the same sequence length.')
    

class SoftmaxAttention(nn.Module):
    """
    Implementation of scaled dot-product attention as described in 
    "Attention Is All You Need" (Vaswani et al., 2017).
    
    This module computes attention scores between query and key tensors, and uses
    these scores to create a weighted sum of value tensors. It supports any number
    of batch dimensions, including configurations where attention heads are represented
    as additional batch dimensions.

    Usage examples:
        >>> # Single-head attention with batch size 32, sequence length 10, dimension 64
        >>> q = torch.randn(32, 10, 64)
        >>> k = torch.randn(32, 15, 64)  # Different sequence length for keys
        >>> v = torch.randn(32, 15, 64)  # Must match key sequence length
        >>> attention = SoftmaxAttention()
        >>> output = attention(q, k, v)  # Shape: (32, 10, 64)
        >>>
        >>> # Multi-head attention with 8 heads, batch size 32, sequences length 10, dimension 64
        >>> # (heads treated as batch dimension)
        >>> q = torch.randn(32, 8, 10, 64)
        >>> k = torch.randn(32, 8, 15, 64)
        >>> v = torch.randn(32, 8, 15, 64)
        >>> output = attention(q, k, v)  # Shape: (32, 8, 10, 64)
    """
    
    def __init__(self, scale: float=None, drop: float=0.):
        """
        Initialize the attention module.
        
        Args:
            scale (float, optional): Custom scaling factor for attention scores.
                If None, uses 1/sqrt(d_k) as in the original paper.
            drop (float, default=0.): Dropout rate applied to attention scores.
        """
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(drop)

    def forward(
            self, 
            q: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor, 
            return_attn=False
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes scaled dot-product attention.
        
        Args:
            q (torch.Tensor): Query tensor of shape (..., Lq, D)
            k (torch.Tensor): Key tensor of shape   (..., Lk, D)
            v (torch.Tensor): Value tensor of shape (..., Lk, Dv)
            return_attn (bool, optional): If True, returns attention weights along with output.
                Default: False
        
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: 
                - If return_attn=False: Output tensor of shape (..., Lq, Dv)
                - If return_attn=True: Tuple of (output tensor, attention weights tensor)
                  where attention weights have shape (..., Lq, Lk)
        """
        _check_attention_dims(q, k, v)
        Datt = q.shape[-1]

        if self.scale is None:
            self.scale = Datt ** (-0.5)
        
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = self.dropout(attn)
        attn = F.softmax(attn, -1)
        out = attn @ v
        if return_attn:
            return out, attn
        return out

class MHSA(nn.Module):
    """
    Multi-Head Self-Attention module as described in "Attention Is All You Need" (Vaswani et al., 2017).
    
    Implements the self-attention mechanism where queries, keys, and values are all 
    projections of the same input sequence. It first projects the input into queries, keys, and values, 
    splits them into multiple heads, applies scaled dot-product attention in parallel, and then 
    recombines the outputs with a final projection.
    
    Usage example:
        >>> # Multi-head self-attention with 8 heads, batch size 32, sequence length 10, dimension 64
        >>> x = torch.randn(32, 10, 64)
        >>> mhsa = MHSA(dim=64, heads=8)
        >>> output = mhsa(x)  # Shape: (32, 10, 64)
    """
    def __init__(self, dim: int, heads: int=8, scale: float=None, drop_out:float = 0., drop_attn: float=0.):
        """
        Initialize the MHSA (Multi-Head Self-Attention) module.
        
        Args:
            dim (int): Input and output dimension.
            heads (int, default=8): Number of attention heads.
            scale (float, optional): Custom scaling factor for attention scores.
                If None, uses 1/sqrt(d_k) as in the original paper.
            drop_out (float, default=0.): Dropout rate applied to the output.
            drop_attn (float, default=0.): Dropout rate applied to attention scores.
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.qkv_projection = nn.Linear(dim, dim*3)
        self.out_projection = nn.Linear(dim, dim)
        self.attention = SoftmaxAttention(scale=scale, drop=drop_attn)
        self.dropout = nn.Dropout(drop_out)
    
    def forward(self, 
                x: torch.Tensor, 
                return_attn: bool=False
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute multi-head self-attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., L, D) where L is the sequence length
                and D is the embedding dimension.
            return_attn (bool, default=False): If True, returns attention weights along with output.
        
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                - If return_attn=False: Output tensor of shape (..., L, D)
                - If return_attn=True: Tuple of (output, attention weights)
                  where attention weights have shape (..., H, L, L) with H being the number of heads
        """
        H, L = self.heads, x.shape[-2]
        
        # Project into H query, key and value sequences
        qkv = self.qkv_projection(x)
        qkv = qkv.reshape(*x.shape[:-2], L, 3, H, -1)  # ... L 3 H D
        q, k, v = qkv.transpose(-4, -2).unbind(-3)       # ... H L D  (3 tensors for q, k, v)

        # Attention
        out = self.attention(q, k, v, return_attn=return_attn)  # ... H L D
        if return_attn:
            out, attn = out

        # Mix outputs from multiple heads
        out = out.transpose(-3, -2).flatten(-2)
        out = self.dropout(out)
        out = self.out_projection(out)

        if return_attn:
            return out, attn
        return out
    

class MHCA(nn.Module):
    """
    Multi-Head Cross-Attention module as described in "Attention Is All You Need" (Vaswani et al., 2017).
    
    This module implements the cross-attention mechanism where queries come from one sequence, 
    while keys and values come from another sequence. It projects the inputs accordingly, splits 
    them into multiple heads, applies scaled dot-product attention in parallel, and then recombines 
    the outputs with a final projection.
        
    Usage example:
        >>> # Multi-head cross-attention with 8 heads, batch size 32, sequence lengths 10 and 15
        >>> x = torch.randn(32, 10, 64)  # Query sequence
        >>> z = torch.randn(32, 15, 64)  # Key/value sequence
        >>> mhca = MHCA(dim=64, heads=8)
        >>> output = mhca(x, z)  # Shape: (32, 10, 64)
    """
    def __init__(self, dim: int, dim_cond: int, heads: int=8, scale: float=None, drop_out:float = 0., drop_attn: float=0.):
        super().__init__()
        """
        Initialize the Multi-Head Cross-Attention module.
        
        Args:
            dim (int): Input and output dimension.
            dim_cond (int): Condition dimension.
            heads (int, default=8): Number of attention heads.
            scale (float, optional): Custom scaling factor for attention scores.
                If None, uses 1/sqrt(d_k) as in the original paper.
            drop_out (float, default=0.): Dropout rate applied to the output.
            drop_attn (float, default=0.): Dropout rate applied to attention scores.
        """
        self.dim = dim
        self.heads = heads
        self.q_projection = nn.Linear(dim, dim)
        self.kv_projection = nn.Linear(dim_cond, 2*dim)
        self.out_projection = nn.Linear(dim, dim)
        self.attention = SoftmaxAttention(scale=scale, drop=drop_attn)
        self.dropout = nn.Dropout(drop_out)
    
    def forward(self, 
                x: torch.Tensor, 
                z: torch.Tensor, 
                return_attn: bool=False
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute multi-head cross-attention.
        
        Args:
            x (torch.Tensor): Query tensor of shape (..., Lx, D) where Lx is the query sequence length
                and D is the embedding dimension.
            z (torch.Tensor): Key/value tensor of shape (..., Lz, D) where Lz is the key/value sequence 
                length and D is the embedding dimension.
            return_attn (bool, default=False): If True, returns attention weights along with output.
        
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                - If return_attn=False: Output tensor of shape (..., Lx, D)
                - If return_attn=True: Tuple of (output tensor, attention weights tensor)
                  where attention weights have shape (..., H, Lx, Lz) with H being the number of heads
        """
        H, Lx, Lz = self.heads, x.shape[-2], z.shape[-2]
        
        # Project into H query, key and value sequences
        q = self.q_projection(x)
        q = q.reshape(*x.shape[:-2], Lx, H, -1) # ... L H D
        q = q.transpose(-3, -2)                 # ... H L D
        
        kv = self.kv_projection(z)
        kv = kv.reshape(*z.shape[:-2], Lz, 2, H, -1)    # ... L 2 H D
        k, v = kv.transpose(-4, -2).unbind(-3)          # ... H L D  (2 tensors: k, v)

        # Attention
        out = self.attention(q, k, v, return_attn=return_attn)  # ... H L D
        if return_attn:
            out, attn = out

        # Mix outputs from multiple heads
        out = out.transpose(-3, -2).flatten(-2)
        out = self.dropout(out)
        out = self.out_projection(out)

        if return_attn:
            return out, attn
        return out
        