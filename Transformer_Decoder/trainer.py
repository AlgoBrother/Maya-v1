import os
import torch
import time

class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, config):
        self.model = model.to(self.device)
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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        # what does weights_only=False do?
        # It loads the entire checkpoint including optimizer and scheduler states, 
        # which is crucial for resuming training without losing momentum or learning rate schedule.
        # If it were True, it would only load the model weights, which might be useful for inference
        # but not for training resumption.
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        print(f"--- Resuming from step {self.step} ---")

    def train(self):
        self.model = torch.compile(self.model)
        self.model.train()
        data_iter = iter(self.train_loader)

        while True:
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _ in range(self.config.grad_accum_steps):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    x, y = next(data_iter)

                x, y = x.to(self.device, non_blocking=True)  # 👈 non_blocking for async transfer
                y = y.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = self.model(x, y)
                    loss = loss / self.config.grad_accum_steps

                loss.backward()  # 👈 no scaler needed for BF16
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            self.step += 1
            
            t1 = time.time()

            # 3. Logging & Checkpointing
            if self.step % 10 == 0:
                dt = (t1 - t0) * 1000 # milliseconds
                print(f"Step {self.step} | Loss: {accum_loss:.4f} | Time: {dt:.2f}ms")

            if self.step % 500 == 0:
                self.save_checkpoint(f"checkpoints/ckpt_step_{self.step}.pt")