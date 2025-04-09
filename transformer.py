import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

from attention import MHSA, MHCA
from feedforward import FeedForward
from position_embeds import SinusoidalPEs

class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention, optional cross-attention for conditioning,
    and feed-forward network with residual connections.
    
    Uses pre-normalization (LayerNorm before each sub-layer) for improved
    training stability as described in https://arxiv.org/pdf/2002.04745.
    
    Args:
        d_model (int): Dimension of the model/embeddings
        heads (int): Number of attention heads
        d_cond (int, optional): Dimension of conditioning input. If None, no cross-attention is used
        drop_attn (float, default=0.): Dropout rate for self-attention.
        drop_xattn (float, default=0.): Dropout rate for cross-attention.
        drop_ffn (float, default=0): Dropout rate for feed-forward network.
    """
    def __init__(
        self, 
        d_model: int, 
        heads: int, 
        d_cond: Optional[int] = None, 
        drop_attn: float = 0., 
        drop_xattn: float = 0., 
        drop_ffn: float = 0.
    ) -> None:
        super().__init__()
        self.self_attn = MHSA(d_model, heads, drop_attn=drop_attn)
        self.norm_sa = nn.LayerNorm(d_model)
        if d_cond is not None:
            self.cross_attn = MHCA(
                d_model, d_cond, heads=heads, drop_attn=drop_xattn
            )
            self.norm_cx = nn.LayerNorm(d_model)
            self.norm_cz = nn.LayerNorm(d_cond)

        self.ffn = FeedForward(d_model, drop=drop_ffn)
        self.norm_ffn = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        x: torch.Tensor, 
        z: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [..., seq_len, d_model]
            z (torch.Tensor, optional): Conditioning tensor for cross-attention
                of shape [..., cond_seq_len, d_cond]. Leading "batch" dimensions 
                should match input.
            mask (torch.Tensor, optional): Optional mask to be applied to 
                self-attention (eg, for padding or causal attention)
                [..., seq_len, seq_len]. Leading "batch" dimensions 
                should match input, or not exist if mask is to be applied to 
                all samples.
                
        Returns:
            torch.Tensor: Output tensor of same shape as input x
        """
        x = x + self.self_attn(self.norm_sa(x), mask=mask)
        if z is not None:
            assert hasattr(self, 'cross_attn')
            x = x + self.cross_attn(self.norm_cx(x), self.norm_cz(z))
        else:
            assert not hasattr(self, 'cross_attn')
        x = x + self.ffn(self.norm_ffn(x))
        return x
    

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder with token embeddings, positional encodings,
    and a stack of transformer blocks.
    
    Args:
        d_model (int): Dimension of the model/embeddings
        depth (int): Number of transformer blocks
        heads (int): Number of attention heads in each transformer block
        vocab (int, default=49408 (CLIPTokenizer)): Vocabulary size for token embeddings.
        d_out (int, optional): Output dimension after final projection. 
                               If None, no projection is applied.
        drop_attn (float, default=0.): Dropout rate for attention.
        drop_ffn (float, default=0.): Dropout rate for feed-forward networks.
    """
    def __init__(
        self, 
        d_model: int, 
        depth: int, 
        heads: int, 
        vocab: int = 49408, 
        d_out: Optional[int] = None, 
        drop_attn: float = 0., 
        drop_ffn: float = 0.
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.tkn_embed = nn.Embedding(vocab, d_model)
        self.pos_embed = SinusoidalPEs(d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, 
                heads, 
                drop_attn=drop_attn, 
                drop_ffn=drop_ffn)
            for _ in range(depth)
        ])

        self.norm = self.fc = None
        if d_out is not None:
            self.norm = nn.LayerNorm(d_model)
            self.fc = nn.Linear(d_model, d_out)

    def forward(self, x_tkns: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer encoder.
        
        Args:
            x_tkns: Input tensor of token indices with shape [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Output tensor with shape [batch_size, seq_len, d_model] 
                (or [batch_size, seq_len, d_out] if d_out is specified)
        """
        L = x_tkns.shape[-1]
        x = self.tkn_embed(x_tkns) + self.pos_embed(L)

        for blk in self.blocks:
            x = blk(x)
        
        if self.fc is not None:
            x = self.fc(self.norm(x))
        
        return x
    

class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_cond: int, 
        depth: int, 
        heads: int, 
        vocab: int = 49408, 
        d_out: Optional[int] = None, 
        drop_attn: float = 0., 
        drop_xattn: float = 0., 
        drop_ffn: float = 0.
    ) -> None:
        """
        Transformer Decoder takes both token inputs and a conditioning tensor,
        allowing it to generate outputs conditioned on external information.
        
        Args:
            d_model (int): Dimension of the model/embeddings
            d_cond (int): Dimension of the conditioning input
            depth (int): Number of transformer blocks
            heads (int): Number of attention heads in each transformer block
            vocab (int, default=49408): Vocabulary size for token embeddings.
            d_out (int, default=None): Output dimension after final projection. 
                If None, no projection is applied.
            drop_attn (float, default=0.): Dropout rate for self-attention.
            drop_xattn (float, default=0.): Dropout rate for cross-attention.
            drop_ffn (float, default=0.): Dropout rate for feed-forward networks.
        """
        super().__init__()
        self.d_model = d_model
        self.tkn_embed = nn.Embedding(vocab, d_model)
        self.pos_embed = SinusoidalPEs(d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, heads, d_cond=d_cond, 
                drop_attn=drop_attn, 
                drop_xattn=drop_xattn, 
                drop_ffn=drop_ffn)
            for _ in range(depth)
        ])

        self.norm = self.fc = None
        if d_out is not None:
            self.norm = nn.LayerNorm(d_model)
            self.fc = nn.Linear(d_model, d_out)

    def forward(self, x_tkns: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer decoder.
        
        Args:
            x_tkns (torch.Tensor): Input tensor of token indices with shape [batch_size, seq_len]
            z (torch.Tensor): Conditioning tensor for cross-attention with shape 
                             [batch_size, cond_seq_len, d_cond]
                
        Returns:
            torch.Tensor: Output tensor with shape [batch_size, seq_len, d_model] 
                         (or [batch_size, seq_len, d_out] if d_out is specified)
        """
        L = x_tkns.shape[-1]
        x = self.tkn_embed(x_tkns) + self.pos_embed(L)
        mask = torch.tril(torch.ones((L, L)))
        
        for blk in self.blocks:
            x = blk(x, z, mask=mask)
        
        if self.fc is not None:
            x = self.fc(self.norm(x))
        
        return x
    
batch_size = 3
inpt_len = 12
cond_len = 3
x_tkns = torch.randint(0, 5000, (batch_size, inpt_len))
z = torch.randn(batch_size, cond_len, 512)
decoder = TransformerDecoder(512, 512, depth=4, heads=8, vocab=5000, drop_attn=0.1, drop_xattn=0.2, drop_ffn=0.1)
output = decoder(x_tkns, z)