"""
Train Epiplexity Guide

Instructions:
To run this training script, first ensure your environment is set up properly.
Source `~/.zshrc` and activate conda env `dllm` (e.g. `conda activate ~/miniconda3/envs/dllm`).

For a single GPU, run:
python examples/epiplexity/train_guide.py

For SLURM with GPU, run:
srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 python examples/epiplexity/train_guide.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dllm.core.models.epiplexity_guide import EpiplexityGuide

class EpiplexityDataset(Dataset):
    """
    Dummy Dataset for Epiplexity Guide Training.
    Replace self.get_item implementation with your actual data generation/loading logic.
    """
    def __init__(self, num_samples=1000, max_seq_len=128, vocab_size=50000):
        super().__init__()
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Dummy data: random tokens and random target structure gains
        input_ids = torch.randint(0, self.vocab_size, (self.max_seq_len,))
        target_gains = torch.rand(self.max_seq_len)
        return input_ids, target_gains

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    vocab_size = 50000
    batch_size = 16
    epochs = 5
    lr = 1e-4

    dataset = EpiplexityDataset(vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EpiplexityGuide(vocab_size=vocab_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0.0
        for input_ids, target_gains in dataloader:
            input_ids = input_ids.to(device)
            target_gains = target_gains.to(device)

            optimizer.zero_grad()
            scores = model(input_ids)
            
            loss = criterion(scores, target_gains)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    train()
