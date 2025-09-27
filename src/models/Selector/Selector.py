import torch
import torch.nn as nn

class Selector(nn.Module):
    def __init__(self, input_dim, output_dim, mode='selection'):
        super(Selector, self).__init__()
        
        hidden_dim = 108

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        ) if mode == 'selection' else nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)
