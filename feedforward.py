import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

class FeedForward(nn.Module):
    """
    A standard feed-forward network module commonly used in transformer architectures.
    
    Args:
        dim (int): Input and output dimension of the module.
        expansion (int, default=4): Expansion factor for the hidden dimension. 
        drop (float, default=0.): Dropout probability for regularization.
    
    Example:
        >>> ff = FeedForward(dim=512, expansion=4, drop=0.1)
        >>> x = torch.randn(batch_size, seq_len, 512)
        >>> output = ff(x)  # Shape: [batch_size, seq_len, 512]
    """
    def __init__(self, dim: int, expansion: int=4, drop: float=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*expansion)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim*expansion, dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [..., dim]
            
        Returns:
            torch.Tensor: Output tensor of the same shape as the input
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)