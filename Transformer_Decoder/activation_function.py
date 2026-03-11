import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MayaConfig

class SwiGLU(nn.Module):
    def __init__(self, config: MayaConfig):
        super(SwiGLU, self).__init__()
        # Expansion factor is typically 2.67x for SwiGLU
        # (2/3 * 4 * n_embed) 
        # n_embed = 768, hidden_dim = 2/3 * 4 * 768 = 2048
        hidden_dim = int((2/3) * 4 * config.n_embd)
        hidden_dim = (hidden_dim + 63) // 64 * 64  # Round to nearest multiple of 64
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)

    def forward(self, x):
        # Swiglu(x) = (Swish(xW1) * xW3) W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
        