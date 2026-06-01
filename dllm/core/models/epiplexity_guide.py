"""
Epiplexity Guide Model

Instructions:
This module contains the `EpiplexityGuide` model, a simple 1-2 layer transformer designed to score the structure gain of tokens.
It is imported and used by `examples/epiplexity/train_guide.py`.

You normally don't run this file directly.
"""

import torch
import torch.nn as nn

class EpiplexityGuide(nn.Module):
    def __init__(self, hidden_size=4096, d_model=256, nhead=8, num_layers=2, max_seq_len=2048):
        super().__init__()
        self.proj_in = nn.Linear(hidden_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output a score for each token
        self.fc_out = nn.Linear(d_model, 1)
        
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size) Float tensor from base LLaDA model
        Returns:
            scores: (batch_size, seq_len) representing the structure gain
        """
        batch_size, seq_len, _ = hidden_states.shape
        positions = torch.arange(0, seq_len, dtype=torch.long, device=hidden_states.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        
        x = self.proj_in(hidden_states) + self.pos_embedding(positions)
        
        # We don't use a causal mask because we're evaluating the unmasking score based on bidirectional canvas features
        x = self.transformer(x)
        scores = self.fc_out(x).squeeze(-1)
        
        return scores
