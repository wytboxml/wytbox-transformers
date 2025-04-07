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
        
        attn_scores = q @ k.transpose(-2, -1) * self.scale
        attn = self.dropout(attn)
        attn = F.softmax(attn, -1)
        out = attn @ v
        if return_attn:
            return out, attn_scores
        return out
