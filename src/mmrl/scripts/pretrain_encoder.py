from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
from rich.console import Console
from rich.table import Table
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from mmrl.representation.market_encoder import MarketPretrainModel

app = typer.Typer(help="Pretrain the CNN/LSTM market encoder.")
console = Console()


class MarketPretrainDataset(Dataset):
    def __init__(self, path: Path):
        data = np.load(path)

        self.order_book = data["order_book"]
        self.aux = data["aux"]
        self.direction = data["direction"]
        self.future_return_bps = data["future_return_bps"]
        self.future_volatility_bps = data["future_volatility_bps"]
        self.future_flow_delta = data["future_flow_delta"]
        self.toxicity = data["toxicity"]

        self.book_channels = int(data["book_channels"][0])
        self.history = int(data["history"][0])
        self.levels = int(data["levels"][0])
        self.aux_dim = int(data["aux_dim"][0])

    def __len__(self) -> int:
        return len(self.direction)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "order_book": torch.from_numpy(self.order_book[idx].astype(np.float32)),
            "aux": torch.from_numpy(self.aux[idx].astype(np.float32)),
            "direction": torch.tensor(int(self.direction[idx]), dtype=torch.long),
            # Scale bps targets so regression magnitudes are civilized.
            "return_scaled": torch.tensor(float(self.future_return_bps[idx]) / 10.0, dtype=torch.float32),
            "volatility_scaled": torch.tensor(float(self.future_volatility_bps[idx]) / 10.0, dtype=torch.float32),
            "flow_delta": torch.tensor(float(self.future_flow_delta[idx]), dtype=torch.float32),
            "toxicity": torch.tensor(float(self.toxicity[idx]), dtype=torch.float32),
        }


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def compute_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    direction_loss = nn.functional.cross_entropy(outputs["direction_logits"], batch["direction"])
    return_loss = nn.functional.smooth_l1_loss(outputs["return_bps"], batch["return_scaled"])
    vol_loss = nn.functional.smooth_l1_loss(outputs["volatility_bps"], batch["volatility_scaled"])
    flow_loss = nn.functional.smooth_l1_loss(outputs["flow_delta"], batch["flow_delta"])
    toxicity_loss = nn.functional.mse_loss(outputs["toxicity"], batch["toxicity"])

    total = (
        direction_loss
        + 0.5 * return_loss
        + 0.25 * vol_loss
        + 0.25 * flow_loss
        + 0.5 * toxicity_loss
    )

    stats = {
        "loss": float(total.item()),
        "direction_loss": float(direction_loss.item()),
        "return_loss": float(return_loss.item()),
        "vol_loss": float(vol_loss.item()),
        "flow_loss": float(flow_loss.item()),
        "toxicity_loss": float(toxicity_loss.item()),
    }

    return total, stats


def json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def print_epoch_table(row: dict[str, Any]) -> None:
    table = Table(title=f"Encoder pretraining epoch {row['epoch']}")
    table.add_column("metric")
    table.add_column("value")

    for key, value in row.items():
        table.add_row(str(key), str(value))

    console.print(table)


@app.command()
def main(
    dataset: Path = typer.Option(Path("data/pretrain/market_pretrain.npz")),
    out: Path = typer.Option(Path("models/market_encoder.pt")),
    epochs: int = typer.Option(10),
    batch_size: int = typer.Option(64),
    lr: float = typer.Option(1e-4),
    val_frac: float = typer.Option(0.2),
    embedding_dim: int = typer.Option(128),
    device_name: str = typer.Option("auto"),
    seed: int = typer.Option(7),
    torch_threads: int = typer.Option(1),
) -> None:
    torch.set_num_threads(max(1, torch_threads))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    if str(device) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MarketPretrainDataset(dataset)

    val_size = max(1, int(len(ds) * val_frac))
    train_size = len(ds) - val_size

    train_ds, val_ds = random_split(
        ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = MarketPretrainModel(
        book_channels=ds.book_channels,
        aux_dim=ds.aux_dim,
        levels=ds.levels,
        embedding_dim=embedding_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    metadata = {
        "dataset": str(dataset),
        "book_channels": ds.book_channels,
        "aux_dim": ds.aux_dim,
        "history": ds.history,
        "levels": ds.levels,
        "embedding_dim": embedding_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": str(device),
    }

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []

        for batch in train_loader:
            batch = batch_to_device(batch, device)

            outputs = model(batch["order_book"], batch["aux"])
            loss, stats = compute_loss(outputs, batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch_to_device(batch, device)
                outputs = model(batch["order_book"], batch["aux"])
                loss, stats = compute_loss(outputs, batch)
                val_losses.append(float(loss.item()))

                pred = outputs["direction_logits"].argmax(dim=1)
                correct += int((pred == batch["direction"]).sum().item())
                total += int(batch["direction"].numel())

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
            "val_direction_acc": correct / max(1, total),
        }

        print_epoch_table(row)

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]

            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": metadata,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                },
                out,
            )

            console.print(f"[green]Saved best encoder: {out}[/green]")

    (out.parent / f"{out.stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=json_default),
        encoding="utf-8",
    )

    console.print({"saved_encoder": str(out), "best_val_loss": best_val_loss})


if __name__ == "__main__":
    app()