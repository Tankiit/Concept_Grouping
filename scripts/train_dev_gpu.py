#!/usr/bin/env python3
"""Run YAML-selected, explicitly non-preregistered GPU CBM variations.

This script does not call certification, audit, or preregistration code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import yaml

from cmcert.dev.cbm import DevCBM
from cmcert.dev.colored_mnist import ColoredMNIST


def accuracy(model: DevCBM, loader: DataLoader, device: torch.device, use_parity_concept: bool = False) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for image, label, _ in loader:
            concept_logits, logits = model(image.to(device, non_blocking=True))
            if use_parity_concept:
                logits = concept_logits[:, 0]
            prediction = logits.gt(0).cpu()
            correct += prediction.eq(label.bool()).sum().item()
            total += len(label)
    return correct / total


def merge(defaults: dict[str, Any], variation: dict[str, Any]) -> dict[str, Any]:
    settings = {**defaults, **variation}
    if settings["label_loss_weight"] == 0 and settings["concept_loss_weight"] == 0:
        raise ValueError("a variation must train at least one loss")
    return settings


def run_variation(name: str, settings: dict[str, Any], device: torch.device) -> None:
    torch.manual_seed(settings["seed"])
    train = ColoredMNIST(settings["data_root"], train=True, rho=settings["rho_train"], seed=settings["seed"], limit=settings["train_size"])
    test_id = ColoredMNIST(settings["data_root"], train=False, rho=settings["rho_id"], seed=settings["seed"] + 1)
    test_shift = ColoredMNIST(settings["data_root"], train=False, rho=settings["rho_shift"], seed=settings["seed"] + 2)
    loader_options = {"num_workers": 4, "pin_memory": True, "persistent_workers": True}
    train_loader = DataLoader(train, settings["batch_size"], shuffle=True, **loader_options)
    id_loader = DataLoader(test_id, settings["batch_size"], **loader_options)
    shift_loader = DataLoader(test_shift, settings["batch_size"], **loader_options)
    model = DevCBM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"], eta_min=settings["min_lr"])
    criterion = nn.BCEWithLogitsLoss()
    use_parity_concept = settings["label_loss_weight"] == 0
    output = Path(settings["output_root"]) / name
    output.mkdir(parents=True, exist_ok=True)
    final_checkpoint = output / "dev_cbm.pt"
    progress_checkpoint = output / "progress.pt"
    if final_checkpoint.is_file():
        print(f"variation={name} already_complete={final_checkpoint}", flush=True)
        return
    start_epoch = 1
    if progress_checkpoint.is_file():
        progress = torch.load(progress_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(progress["state_dict"])
        optimizer.load_state_dict(progress["optimizer_state"])
        scheduler.load_state_dict(progress["scheduler_state"])
        start_epoch = progress["epoch"] + 1
        print(f"variation={name} resuming_epoch={start_epoch}", flush=True)
    with SummaryWriter(log_dir=str(output / "tensorboard")) as writer:
        writer.add_text("run/settings", yaml.safe_dump(settings, sort_keys=True), 0)
        for epoch in range(start_epoch, settings["epochs"] + 1):
            model.train()
            total_loss = 0.0
            progress = tqdm(
                train_loader,
                desc=f"{name} {epoch}/{settings['epochs']}",
                leave=False,
                dynamic_ncols=True,
                disable=not sys.stderr.isatty(),
            )
            for image, label, concepts in progress:
                image = image.to(device, non_blocking=True)
                label, concepts = label.float().to(device), concepts.float().to(device)
                concept_logits, prediction_logits = model(image)
                loss = (
                    settings["label_loss_weight"] * criterion(prediction_logits, label)
                    + settings["concept_loss_weight"] * criterion(concept_logits, concepts)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(label)
                progress.set_postfix(loss=f"{loss.item():.3f}")
            scheduler.step()
            writer.add_scalar("train/loss", total_loss / len(train), epoch)
            writer.add_scalar("optimization/lr", scheduler.get_last_lr()[0], epoch)
            if epoch % settings["checkpoint_every"] == 0 or epoch == settings["epochs"]:
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                    },
                    progress_checkpoint,
                )
            if epoch % settings["log_every"] == 0 or epoch == 1 or epoch == settings["epochs"]:
                id_accuracy = accuracy(model, id_loader, device, use_parity_concept)
                shift_accuracy = accuracy(model, shift_loader, device, use_parity_concept)
                writer.add_scalar("eval/id_accuracy", id_accuracy, epoch)
                writer.add_scalar("eval/shift_accuracy", shift_accuracy, epoch)
                print(
                    f"variation={name} epoch={epoch} loss={total_loss / len(train):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} id_accuracy={id_accuracy:.4f} "
                    f"shift_accuracy={shift_accuracy:.4f}",
                    flush=True,
                )
        metrics = {
            "id_accuracy": accuracy(model, id_loader, device, use_parity_concept),
            "shift_accuracy": accuracy(model, shift_loader, device, use_parity_concept),
        }
        writer.add_hparams(
            {key: value for key, value in settings.items() if isinstance(value, (str, int, float, bool))},
            {f"final/{key}": value for key, value in metrics.items()},
        )
    (output / "metrics.yaml").write_text(yaml.safe_dump(metrics, sort_keys=True), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "variation": name, "settings": settings, "metrics": metrics}, final_checkpoint)
    print(f"variation={name} saved={final_checkpoint} metrics={metrics}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dev_variations.yaml")
    parser.add_argument("--variation", action="append", help="Variation name; repeat it, or omit to run all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this requested GPU smoke test")
    device = torch.device("cuda")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    defaults, variations = config["defaults"], config["variations"]
    selected = args.variation or list(variations)
    unknown = set(selected).difference(variations)
    if unknown:
        raise ValueError(f"unknown variations: {sorted(unknown)}")
    for number, name in enumerate(selected):
        settings = merge(defaults, variations[name])
        settings["seed"] += number
        run_variation(name, settings, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
