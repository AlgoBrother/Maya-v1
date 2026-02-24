import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNormalisation(nn.Module):
    def __init__(self, embed_dim, eps = 1e-8):
        super(RMSNormalisation, self).__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))
        
    def forward(self, x):
        #x : (batch, seq_len, dim)
        x_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_norm * self.weight
    
        
