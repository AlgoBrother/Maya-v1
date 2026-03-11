import os
import torch
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import MayaConfig
from model import MayaTransformer
from dataloader import get_dataloader
from trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to tokenized .bin data directory")
    args = parser.parse_args()

    # ── 1. Config ──────────────────────────────────────────────────────────────
    config = MayaConfig(
        block_size      = 1024,
        vocab_size      = 50304,
        n_layers        = 12,
        n_heads         = 8,
        n_embd          = 768,
        dropout         = 0.0,
        bias            = False,
        batch_size      = 4,
        grad_accum_steps= 8,
        learning_rate   = 6e-4,
        weight_decay    = 0.1,
    ) # I love doing extra work because I am scared of my own code ;-; if it works, it works.
    # I could have just called MayaConfig() without args and it would have used the defaults, but I wanted to be explicit here for clarity.
    # Also, this way I can easily modify the config values in one place without having to scroll through the code to find where they are set. 
    # It's a bit of extra work upfront, but it pays off in the long run when I want to experiment with different hyperparameters.
    # [The lines after "MayaConfig()" was done by my copilot and the reason is cool so imma stick with it. :] ]

    # ── 2. Model ───────────────────────────────────────────────────────────────
    model = MayaTransformer(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Maya model | {num_params / 1e6:.2f}M parameters")

    # ── 3. Optimizer ───────────────────────────────────────────────────────────
    # Separate weight decay: don't decay biases or norm weights
    decay_params     = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if p.dim() < 2]

    optimizer = AdamW([
        {"params": decay_params,    "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)

    # ── 4. Scheduler: Linear Warmup + Cosine Annealing ────────────────────────
    warmup_steps  = 100
    total_steps   = 10_000  # adjust to your dataset size

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps                          # linear warmup
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))  # cosine decay to 10% of learning rate

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 5. Dataloader ──────────────────────────────────────────────────────────
    train_loader = get_dataloader(args.data_dir, config)

    # ── 6. Trainer ─────────────────────────────────────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)

    trainer = Trainer(
        model        = model,
        optimizer    = optimizer,
        scheduler    = scheduler,
        train_loader = train_loader,
        config       = config,
    )

    # ── 7. Resume or start fresh ───────────────────────────────────────────────
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    else:
        print("Starting fresh training run")

    # ── 8. Train ───────────────────────────────────────────────────────────────
    trainer.train()

if __name__ == "__main__":
    main()