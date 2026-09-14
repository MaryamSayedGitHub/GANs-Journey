"""
01_VAE/train.py
----------------
سكريبت تدريب الـ VAE على MNIST.

تشغيل:
    python train.py --epochs 20 --batch-size 128 --latent-dim 20
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_device, get_mnist_dataloader, save_sample_grid, plot_losses  # noqa: E402
from model import VAE, vae_loss_function  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    dataloader = get_mnist_dataloader(batch_size=args.batch_size)
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"Total": [], "Reconstruction": [], "KL": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_recon, epoch_kl = 0.0, 0.0, 0.0

        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, _ in loop:
            images = images.view(images.size(0), -1).to(device)

            x_hat, mu, log_var = model(images)
            loss, recon, kl = vae_loss_function(x_hat, images, mu, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            loop.set_postfix(loss=loss.item() / images.size(0))

        n = len(dataloader.dataset)
        history["Total"].append(epoch_loss / n)
        history["Reconstruction"].append(epoch_recon / n)
        history["KL"].append(epoch_kl / n)

        # حفظ عينات كل epoch عشان نتابع التقدم
        model.eval()
        samples = model.sample(64, device).view(-1, 1, 28, 28).cpu()
        save_sample_grid(samples, f"{args.output_dir}/samples/epoch_{epoch:03d}.png")

    plot_losses(history, f"{args.output_dir}/loss_curve.png", title="VAE Training Loss")
    torch.save(model.state_dict(), f"{args.output_dir}/vae_final.pth")
    print("تم التدريب وحفظ الموديل بنجاح.")


if __name__ == "__main__":
    main()
