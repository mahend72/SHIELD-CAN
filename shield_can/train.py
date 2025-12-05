# train.py
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from shield_can.config import FeatureConfig, ModelConfig, TrainingConfig
from shield_can.dataset import CANWindowDataset
from shield_can.model import EdgeTransformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="runs/shield_can")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    feat_cfg = FeatureConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig(
        batch_size=args.batch_size, num_epochs=args.epochs, lr=args.lr, device=device
    )

    # label mapping: adjust as per your dataset
    # e.g.: {"Normal": 0, "DoS": 1, "Fuzzy": 2, "Malfunction": 3, "Spoof": 4}
    label_map = {
        "Normal": 0,
        "DoS": 1,
        "Fuzzy": 2,
        "Malfunction": 3,
        "Spoof": 4,
    }

    train_ds = CANWindowDataset(
        args.train_csv, feat_cfg=feat_cfg, model_cfg=model_cfg, label_map=label_map
    )
    val_ds = CANWindowDataset(
        args.val_csv, feat_cfg=feat_cfg, model_cfg=model_cfg, label_map=label_map
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = EdgeTransformer(model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg.num_epochs)

    best_val_f1 = 0.0

    for epoch in range(1, train_cfg.num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total

        # validation
        model.eval()
        from sklearn.metrics import f1_score

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(y.cpu().numpy().tolist())

        macro_f1 = f1_score(all_targets, all_preds, average="macro")

        print(
            f"Epoch {epoch}/{train_cfg.num_epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_macro_f1={macro_f1:.4f}"
        )

        # save best
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            ckpt_path = out_dir / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path} (macro F1={best_val_f1:.4f})")


if __name__ == "__main__":
    main()
