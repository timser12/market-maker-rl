from __future__ import annotations

import torch
from torch import nn


class OrderBookCNNEncoder(nn.Module):
    """CNN encoder over [channels, time, levels]."""

    def __init__(self, in_channels: int, embedding_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 16)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 16, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingDQN(nn.Module):
    """Dueling DQN with CNN market-depth branch and auxiliary feature branch."""

    def __init__(self, book_channels: int, portfolio_dim: int, n_actions: int):
        super().__init__()

        self.book_encoder = OrderBookCNNEncoder(book_channels, 128)

        self.aux_encoder = nn.Sequential(
            nn.Linear(portfolio_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        combined_dim = 128 + 64

        self.value = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, order_book: torch.Tensor, portfolio: torch.Tensor) -> torch.Tensor:
        book_emb = self.book_encoder(order_book)
        aux_emb = self.aux_encoder(portfolio)

        x = torch.cat([book_emb, aux_emb], dim=-1)

        value = self.value(x)
        advantage = self.advantage(x)

        return value + advantage - advantage.mean(dim=-1, keepdim=True)