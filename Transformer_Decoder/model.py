import torch
import torch.nn as nn
from torch.nn import functional as F
from config import MayaConfig
from multi_head_attention import Block, RoPE
from activation_function import SwiGLU
from rms_norm import RMSNormalisation
import math

class MayaTransformer(nn.Module):
    def __init__(self, config: MayaConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = RMSNormalisation(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying (Production Standard for memory savings)
        self.transformer.wte.weight = self.lm_head.weight 

        # Initialize weights
        self.apply(self._init_weights)
        
        self.rope = RoPE(config.n_embd // config.n_heads) # head_dim = n_embd // n_heads 
        # what is head_dim?
        # head_dim is the dimension of each attention head in a multi-head attention mechanism.
        # It is calculated as the total embedding dimension (n_embd) divided by the number 
        # of attention heads (n_heads).
        # This allows the model to split the embedding into multiple heads for parallel attention computations,
        # which can capture different aspects of the input data.
        
        # Apply special scaled init to output projections (see GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        # Token embeddings
        x = self.transformer.wte(idx)
        cos, sin = self.rope(x)  # Precompute RoPE embeddings for the sequence length
        
        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x, cos, sin)  # Passing RoPE embeddings to each block
            
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If we are training, calculate loss here for efficiency
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference mode: only return last logit for speed
            logits = self.lm_head(x[:, [-1], :]) 
            loss = None

        return logits, loss