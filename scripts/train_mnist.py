#!/usr/bin/env python3
"""Train assignment Part B models and export checkpoints, curves, and grids."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flow_models.models import MNISTUNet, SingleStepDenoiser  # noqa: E402


def loader(batch: int, root: Path) -> tuple[DataLoader, DataLoader]:
    tfm = transforms.ToTensor()
    train = datasets.MNIST(root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root, train=False, download=True, transform=tfm)
    return (DataLoader(train, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(test, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True))


def save_curve(losses: list[float], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(losses, color="#111111", linewidth=1.5)
    ax.set(title=title, xlabel="Training step", ylabel="MSE loss")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


@torch.no_grad()
def sample_flow(model: MNISTUNet, device: torch.device, n: int = 64, labels: torch.Tensor | None = None,
                steps: int = 100, guidance: float = 0.0) -> torch.Tensor:
    x = torch.randn(n, 1, 28, 28, device=device)
    for i in range(steps):
        t = i / steps
        h = 1 / steps
        tv = torch.full((n,), t, device=device)
        if model.class_conditioned and guidance != 0 and labels is not None:
            vc = model(x, tv, labels)
            vu = model(x, tv, torch.full_like(labels, -1))
            v = vu + guidance * (vc - vu)
        else:
            v = model(x, tv, labels)
        x = x + h * v
    return x.clamp(0, 1)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    train_dl, test_dl = loader(args.batch_size, Path(args.data))
    torch.manual_seed(args.seed)

    # Assignment noising visualization, reusing the same clean test digits.
    x_demo, _ = next(iter(test_dl)); x_demo = x_demo[:8].to(device)
    sigmas = [0.0, .2, .4, .5, .6, .8, 1.0]
    noisy_rows = [x_demo]
    for sigma in sigmas:
        noisy_rows.append((x_demo + sigma * torch.randn_like(x_demo)).clamp(0, 1))
    save_image(torch.cat(noisy_rows), out / "noise-process.png", nrow=8)

    # The assignment has two denoising objectives: additive noise and pure noise.
    for pure_noise in (False, True):
        tag = "pure-noise-denoiser" if pure_noise else "denoiser"
        denoiser = SingleStepDenoiser(args.denoiser_dim).to(device)
        opt = torch.optim.Adam(denoiser.parameters(), lr=1e-4)
        denoise_losses: list[float] = []
        for epoch in range(args.denoiser_epochs):
            denoiser.train()
            for x, _ in train_dl:
                x = x.to(device)
                z = torch.randn_like(x) if pure_noise else x + 0.5 * torch.randn_like(x)
                loss = nn.functional.mse_loss(denoiser(z), x)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                denoise_losses.append(loss.item())
            denoiser.eval()
            x, _ = next(iter(test_dl)); x = x[:64].to(device)
            z = torch.randn_like(x) if pure_noise else x + 0.5 * torch.randn_like(x)
            save_image(torch.cat([z.clamp(0, 1), denoiser(z).clamp(0, 1)]), out / f"{tag}-epoch-{epoch+1}.png", nrow=8)
        torch.save(denoiser.state_dict(), out / f"{tag}.pt")
        save_curve(denoise_losses, out / f"{tag}-loss.png", tag.replace('-', ' ').title())
        if not pure_noise:
            # OOD noise sweep on the same fixed test digit, with a fresh epsilon per level.
            denoiser.eval()
            source = x_demo[:1].expand(len(sigmas), -1, -1, -1)
            eps = torch.randn_like(source)
            sweep = torch.stack([(source[i:i+1] + s * eps[i:i+1]).squeeze(0) for i, s in enumerate(sigmas)])
            pred = denoiser(sweep).clamp(0, 1)
            save_image(torch.cat([source, sweep.clamp(0, 1), pred]), out / "denoiser-ood-sweep.png", nrow=8)

    for conditioned in (False, True):
        model = MNISTUNet(args.flow_dim, class_conditioned=conditioned).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / args.flow_epochs)) if args.scheduler else None
        losses: list[float] = []
        name = "class-flow" if conditioned else "time-flow"
        for epoch in range(args.flow_epochs):
            model.train()
            for x1, labels in train_dl:
                x1 = x1.to(device); labels = labels.to(device)
                x0 = torch.randn_like(x1); t = torch.rand(x1.shape[0], device=device)
                xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
                target = x1 - x0
                cond = labels.clone() if conditioned else None
                if conditioned:
                    cond[torch.rand_like(t) < 0.1] = -1
                pred = model(xt, t, cond)
                loss = nn.functional.mse_loss(pred, target)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                losses.append(loss.item())
            if scheduler: scheduler.step()
            model.eval()
            if conditioned:
                y = torch.arange(10, device=device).repeat_interleave(4)
                samples = sample_flow(model, device, n=40, labels=y, guidance=5.0, steps=args.sample_steps)
                save_image(samples, out / f"{name}-epoch-{epoch+1}.png", nrow=10)
            elif epoch + 1 in {1, 5, 10}:
                samples = sample_flow(model, device, n=64, steps=args.sample_steps)
                save_image(samples, out / f"{name}-epoch-{epoch+1}.png", nrow=8)
        torch.save(model.state_dict(), out / f"{name}.pt")
        save_curve(losses, out / f"{name}-loss.png", f"{name.replace('-', ' ').title()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=None, help="cpu, cuda, or auto")
    p.add_argument("--data", default="data"); p.add_argument("--output", default="outputs")
    p.add_argument("--seed", type=int, default=180)
    p.add_argument("--denoiser-epochs", type=int, default=5); p.add_argument("--flow-epochs", type=int, default=10)
    p.add_argument("--denoiser-dim", type=int, default=128); p.add_argument("--flow-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256); p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--scheduler", action=argparse.BooleanOptionalAction, default=True)
    train(p.parse_args())


if __name__ == "__main__": main()
