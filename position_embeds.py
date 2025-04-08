import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

class SinusoidalPEs(nn.Module):
    """
    Sinusoidal Positional Encodings module as described in 'Attention Is All You Need'.
        
    Args:
        dim (int): The embedding dimension. Should be divisible by 2.
    
    Attributes:
        omega (torch.Tensor): Frequency tensor with shape (1, dim//2) containing
            frequencies that decrease exponentially from 1 to 1/10000.
    
    Example:
        >>> pos_encoder = SinusoidalPEs(dim=512)
        >>> # Get positional encodings for positions 0 to 99
        >>> pos_embeddings = pos_encoder(L=100)  # shape: (100, 512)
        >>> # Or with custom position indices
        >>> positions = torch.tensor([[0], [10], [100]])  # shape: (3, 1)
        >>> pos_embeddings = pos_encoder(pos=positions)  # shape: (3, 512)
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        omega = torch.exp(- math.log(10000) * torch.arange(0, dim, 2) / dim)    # D//2
        self.register_buffer('omega', omega[None])  # 1 D//2
        
    def forward(self, L: int=None, pos: torch.Tensor=None) -> torch.Tensor:
        """
        Compute sinusoidal positional encodings for the given positions.
        
        Args:
            L (int, optional): The sequence length for which to generate positional
                encodings. Only used when pos is None.
            pos (torch.Tensor, optional): A tensor of positions to encode.
                If None, positions are generated using the provided L parameter.
                
        Returns:
            torch.Tensor: Positional encodings with shape (..., dim), where the
                leading dimensions match those of the input pos tensor, or (L, dim)
                if positions were generated using L.
                
        Note:
            Either pos or L must be provided, but not both.
        """
        assert pos is not None or L is not None
        
        if pos is None:
            pos = torch.arange(L)
        pos = pos.unsqueeze(-1)     # ... L 1
        
        pos_embeds = torch.cat((
            torch.sin(pos * self.omega),
            torch.cos(pos * self.omega)
            ), dim=-1)

        return pos_embeds

