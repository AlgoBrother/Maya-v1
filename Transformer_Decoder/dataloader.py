import os
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from config import MayaConfig

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
            chunk = torch.load(file_path, weights_only=True)
            data = chunk['tokens']                   
            print("Token count:", len(data))

            num_tokens = len(data)
            num_samples = (num_tokens - 1) // self.block_size

            idxs = list(range(num_samples))
            if self.shuffle:
                random.shuffle(idxs)

            for i in idxs:
                start = i * self.block_size
                end = start + self.block_size
                x = data[start:end].long()
                y = data[start+1:end+1].long()
                yield x, y

def get_dataloader(data_dir, config: MayaConfig):
    dataset = MayaStreamingDataset(data_dir, config.block_size)
    # pin_memory=True is critical for fast GPU transfer on your 4070
    return DataLoader(dataset, batch_size=config.batch_size, pin_memory=True, num_workers=1)