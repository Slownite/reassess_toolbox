from torch import nn
import torch

class STV(nn.Module):
    def __init__(self, rgb_embed_dim, flow_emded_dim, n_present_heads=1, n_temporal_heads=1):
        super().__init__(self)
        self.present_heads = nn.MultiheadAttention(rgb_embed_dim + flow_emded_dim, n_present_heads, dropout=0.5)
        self.temporal_heads = nn.MultiHeadAttention(r)
        
