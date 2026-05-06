from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class MarketEncoderCNNLSTM(nn.Module):
    """
    Encodes order-book history + auxiliary market state into a compact embedding.

    Input:
        order_book: [batch, channels, time, levels]
        aux:        [batch, aux_dim]

    Output:
        embedding:  [batch, embedding_dim]
    """

    def __init__(
        self,
        book_channels: int,
        aux_dim: int,
        levels: int,
        cnn_dim: int = 128,
        lstm_hidden_dim: int = 128,
        aux_hidden_dim: int = 64,
        embedding_dim: int = 128,
        lstm_layers: int = 1,
    ):
        super().__init__()

        self.book_channels = int(book_channels)
        self.aux_dim = int(aux_dim)
        self.levels = int(levels)
        self.embedding_dim = int(embedding_dim)

        self.level_cnn = nn.Sequential(
            nn.Conv1d(book_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, cnn_dim),
            nn.LayerNorm(cnn_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, aux_hidden_dim),
            nn.LayerNorm(aux_hidden_dim),
            nn.ReLU(),
            nn.Linear(aux_hidden_dim, aux_hidden_dim),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden_dim + aux_hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

    def forward(self, order_book: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        # order_book: [B, C, T, L]
        b, c, t, l = order_book.shape

        if c != self.book_channels:
            raise RuntimeError(f"Expected {self.book_channels} book channels, got {c}")

        x = order_book.permute(0, 2, 1, 3).contiguous()  # [B, T, C, L]
        x = x.view(b * t, c, l)  # [B*T, C, L]

        cnn_features = self.level_cnn(x)  # [B*T, cnn_dim]
        cnn_features = cnn_features.view(b, t, -1)  # [B, T, cnn_dim]

        lstm_out, _ = self.lstm(cnn_features)
        temporal_embedding = lstm_out[:, -1, :]  # last time step

        aux_embedding = self.aux_encoder(aux)

        return self.fusion(torch.cat([temporal_embedding, aux_embedding], dim=-1))


class MarketPretrainModel(nn.Module):
    """
    Multi-task pretrained market model.

    Predicts:
        direction logits:      down / flat / up
        future return bps:     regression
        realized vol bps:      regression
        future flow delta:     regression
        toxicity score:        regression in [0, 1]
    """

    def __init__(
        self,
        book_channels: int,
        aux_dim: int,
        levels: int,
        embedding_dim: int = 128,
    ):
        super().__init__()

        self.encoder = MarketEncoderCNNLSTM(
            book_channels=book_channels,
            aux_dim=aux_dim,
            levels=levels,
            embedding_dim=embedding_dim,
        )

        self.direction_head = nn.Linear(embedding_dim, 3)
        self.return_head = nn.Linear(embedding_dim, 1)
        self.volatility_head = nn.Linear(embedding_dim, 1)
        self.flow_delta_head = nn.Linear(embedding_dim, 1)
        self.toxicity_head = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, order_book: torch.Tensor, aux: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(order_book, aux)

        return {
            "embedding": z,
            "direction_logits": self.direction_head(z),
            "return_bps": self.return_head(z).squeeze(-1),
            "volatility_bps": self.volatility_head(z).squeeze(-1),
            "flow_delta": self.flow_delta_head(z).squeeze(-1),
            "toxicity": self.toxicity_head(z).squeeze(-1),
        }


class FrozenEncoderDQN(nn.Module):
    """
    DQN policy using a frozen pretrained market encoder.

    The CNN/LSTM encoder produces z_t.
    A small dueling MLP learns Q-values from z_t.
    """

    def __init__(
        self,
        encoder: MarketEncoderCNNLSTM,
        embedding_dim: int,
        n_actions: int,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.value = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, order_book: torch.Tensor, portfolio: torch.Tensor) -> torch.Tensor:
        if not any(p.requires_grad for p in self.encoder.parameters()):
            with torch.no_grad():
                z = self.encoder(order_book, portfolio)
        else:
            z = self.encoder(order_book, portfolio)

        value = self.value(z)
        advantage = self.advantage(z)

        return value + advantage - advantage.mean(dim=-1, keepdim=True)


def load_pretrained_encoder(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[MarketEncoderCNNLSTM, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    metadata = checkpoint["metadata"]

    model = MarketPretrainModel(
        book_channels=int(metadata["book_channels"]),
        aux_dim=int(metadata["aux_dim"]),
        levels=int(metadata["levels"]),
        embedding_dim=int(metadata["embedding_dim"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model.encoder, metadata