import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, config):
        super(SwiGLU, self).__init__()
        # Expansion factor is typically 2.67x for SwiGLU
        # (2/3 * 4 * n_embed) 
        # n_embed = 512 
        # hidden_dim = 2.67 * n_embed = 2.67 * 512 = 1365.44 ~ 1365
        # rounding to 1408 for better GPU performance (multiple of 64)
        hidden_dim = 1408
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)

    def forward(self, x):
        # Swiglu(x) = (Swish(xW1) * xW3) W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
        