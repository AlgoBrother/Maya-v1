import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rms_norm import RMSNormalisation
from config import MayaConfig
from activation_function import SwiGLU

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # returns cos and sin embeddings
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]
    
def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# Shape is (B, T, n_head, head_dim)
# where B is batch size, T is sequence length, n_head is number of attention heads, and head_dim is the dimension of each head (embed_dim // num_heads)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: MayaConfig):
        super().__init__()
        self.n_head = config.n_heads
        self.n_embd = config.n_embd
        # Key, Query, Value in one batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.head_dim = config.n_embd // config.n_heads

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rope(q, k, cos, sin)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        # Flash Attention (Fast & Memory Efficient)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config: MayaConfig):
        super().__init__()
        self.ln_1 = RMSNormalisation(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNormalisation(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x