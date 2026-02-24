import os
import torch
import time
from torch.nn.parallel import DistributedDataParallel as DDP

class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler() # For Mixed Precision (bf16/fp16)
        self.step = 0

    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"--- Checkpoint saved at step {self.step} ---")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        print(f"--- Resuming from step {self.step} ---")

    def train(self):
        self.model.to(self.device)
        self.model.train()
        
        # We use an iterator because our dataset is an IterableDataset (streaming)
        data_iter = iter(self.train_loader)
        
        while True:
            t0 = time.time()
            
            # 1. Gradient Accumulation Loop
            self.optimizer.zero_grad(set_to_none=True) # More efficient than zero_grad()
            accum_loss = 0.0
            
            for _ in range(self.config.grad_accum_steps):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader) # Cycle through your 75 chunks
                    x, y = next(data_iter)
                
                x, y = x.to(self.device), y.to(self.device)
                
                # Mixed Precision Context
                with torch.amp.autocast(dtype=torch.bfloat16):
                    logits, loss = self.model(x, y)
                    loss = loss / self.config.grad_accum_steps # Normalize for accumulation
                
                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

            # 2. Optimization Step
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Prevent exploding gradients
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            self.step += 1
            t1 = time.time()

            # 3. Logging & Checkpointing
            if self.step % 10 == 0:
                dt = (t1 - t0) * 1000 # milliseconds
                print(f"Step {self.step} | Loss: {accum_loss:.4f} | Time: {dt:.2f}ms")

            if self.step % 500 == 0:
                self.save_checkpoint(f"checkpoints/ckpt_step_{self.step}.pt")