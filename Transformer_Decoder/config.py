from dataclasses import dataclass

@dataclass
class MayaConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 8
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    batch_size: int = 4
    
    grad_accum_steps: int = 8 
    # what is grad_accum_steps?
    # grad_accum_steps is the number of steps to accumulate gradients before performing a backward pass and optimizer step.
    # This is useful when training with large batch sizes that may not fit in memory.
    
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    # what is weight_decay?
    # weight_decay is a regularization technique that adds a penalty to the loss function based on the magnitude of the model's weights.
    # This helps prevent overfitting by discouraging the model from relying too heavily on any particular feature or set of features.
    