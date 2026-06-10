import os
import torch
import time
from datetime import timedelta
from config import MayaConfig

class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, config: MayaConfig):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.config = config
        self.step = 0
        self.best_loss = float('inf')

        # Timing buffers for moving average
        self.window_size = 20
        self.step_times = []

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'best_loss': self.best_loss,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"--- Checkpoint saved at step {self.step} ---")
        self._cleanup_checkpoints(keep_last=3)
        
    
    def _cleanup_checkpoints(self, keep_last=3):
        import glob
        checkpoint_files = sorted(glob.glob("/mnt/d/Maya_checkpoints/ckpt_step_*.pt"), key=os.path.getmtime)
        for old_ckpt in checkpoint_files[:-keep_last]: # delete all but the last `keep_last` checkpoints
            os.remove(old_ckpt)
            print(f"--- Old checkpoint removed: {old_ckpt} ---")

    # def load_checkpoint(self, path):
    #     checkpoint = torch.load(path, map_location=self.device, weights_only=False)
    #     state_dict = checkpoint['model_state_dict']
    #     unwrapped = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
    #     self.model.load_state_dict(unwrapped)
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     self.step = checkpoint['step']
    #     self.best_loss = checkpoint.get('best_loss', float('inf'))  # restore best loss
    #     print(f"--- Resuming from step {self.step} | Best loss: {self.best_loss:.4f} ---")
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        unwrapped = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(unwrapped)

        # Guard against None optimizer state (re-embedding checkpoint)
        if checkpoint.get('optimizer_state_dict') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.step = checkpoint['step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"--- Resuming from step {self.step} | Best loss: {self.best_loss:.4f} ---")

    def train(self, total_steps=192000):
        # Compile model for speed (requires PyTorch 2.0+)
        if not hasattr(self, '_compiled'):
            print("--- Compiling model... ---")
            self.model = torch.compile(self.model)
            self._compiled = True

        self.model.train()
        data_iter = iter(self.train_loader)
        last_best_save_step = self.step

        print(f"--- Training started. Target: {total_steps} steps ---")

        while self.step < total_steps:
            t0 = time.time()

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            # Gradient Accumulation Loop
            for _ in range(self.config.grad_accum_steps):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    x, y = next(data_iter)

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = self.model(x, y)
                    loss = loss / self.config.grad_accum_steps

                loss.backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            torch.cuda.synchronize()
            t1 = time.time()

            dt = t1 - t0
            self.step += 1

            # Moving average for stable ETC
            self.step_times.append(dt)
            if len(self.step_times) > self.window_size:
                self.step_times.pop(0)
            avg_dt = sum(self.step_times) / len(self.step_times)

            # Logging & ETC
            if self.step % 10 == 0:
                steps_left = total_steps - self.step
                etc_seconds = steps_left * avg_dt
                etc_str = str(timedelta(seconds=int(etc_seconds)))

                tokens_per_sec = (self.config.batch_size * self.config.block_size *
                                  self.config.grad_accum_steps) / dt
                print(f"Step {self.step}/{total_steps} | Loss: {accum_loss:.4f} | "
                      f"Time: {dt*1000:.0f}ms | Tok/s: {tokens_per_sec:,.0f} | ETC: {etc_str}")

            # Best checkpoint - only after step 100, check every 25 steps
            if self.step % 25 == 0 and self.step > 100:
                if accum_loss < self.best_loss:
                    improvement = (self.best_loss - accum_loss) / (self.best_loss + 1e-9)
                    if improvement > 0.01 or (self.step - last_best_save_step) >= 100:
                        self.best_loss = accum_loss
                        self.save_checkpoint("/mnt/d/Maya_checkpoints/best.pt")
                        print(f"--- New Best Model Saved (Step {self.step}, Loss: {self.best_loss:.4f}) ---")
                        last_best_save_step = self.step

            # Regular checkpoint every 1000 steps
            if self.step % 1000 == 0:
                self.save_checkpoint(f"/mnt/d/Maya_checkpoints/ckpt_step_{self.step}.pt")
                print(f"--- Regular Checkpoint Saved (Step {self.step}) ---")

        print(f"--- Training complete! Final loss: {accum_loss:.4f} ---")