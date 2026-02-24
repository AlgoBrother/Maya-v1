import os
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

class MayaStreamingDataset(IterableDataset):
    def __init__(self, data_dir, block_size, shuffle=True):
        self.block_size = block_size
        self.shuffle = shuffle
        
        # Get all chunk files (0-74)
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.bin')])
        print("Data dir:", data_dir)
        print("Files found:", self.files)
        
    def __iter__(self):
        # Master move: Shuffle the file list every epoch
        if self.shuffle:
            random.shuffle(self.files)
            
        for file_path in self.files:
            print("Opening:", file_path)
            # 'r' mode with np.memmap is the secret to low RAM usage
            data = np.memmap(file_path, dtype=np.uint16, mode='r')
            print("Token count:", len(data))
            
            # Calculate how many full sequences we can get
            num_tokens = len(data)
            num_samples = (num_tokens - 1) // self.block_size
            
            # Create a list of indices and shuffle them for within-chunk variety
            idxs = list(range(num_samples))
            if self.shuffle:
                random.shuffle(idxs)
                
            for i in idxs:
                start = i * self.block_size
                end = start + self.block_size
                
                # Convert to tensor. .astype(np.int64) is needed for PyTorch CrossEntropy
                x = torch.from_numpy(data[start:end].astype(np.int64))
                y = torch.from_numpy(data[start+1:end+1].astype(np.int64))
                
                yield x, y

def get_dataloader(data_dir, config):
    dataset = MayaStreamingDataset(data_dir, config.block_size)
    # pin_memory=True is critical for fast GPU transfer on your 4070
    return DataLoader(dataset, batch_size=4, pin_memory=True, num_workers=1)