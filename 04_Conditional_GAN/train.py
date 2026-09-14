"""
04_Conditional_GAN/train.py
-----------------------------
تدريب cGAN على MNIST مع القدرة على توليد رقم محدد عند الطلب.

تشغيل:
    python train.py --epochs 50 --batch-size 128
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_device, get_mnist_dataloader, save_sample_grid, plot_losses  # noqa: E402
from model import Generator, Discriminator  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    dataloader = get_mnist_dataloader(batch_size=args.batch_size)

    G = Generator(latent_dim=args.latent_dim, num_classes=args.num_classes).to(device)
    D = Discriminator(num_classes=args.num_classes).to(device)

    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    history = {"Generator": [], "Discriminator": []}

    # نويز وليبلز ثابتين عشان نتابع تطور كل رقم من 0 لـ 9 عبر الـ epochs
    fixed_noise = torch.randn(args.num_classes * 8, args.latent_dim, device=device)
    fixed_labels = torch.arange(args.num_classes, device=device).repeat_interleave(8)

    for epoch in range(1, args.epochs + 1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in loop:
            batch_size = images.size(0)
            real = images.view(batch_size, -1).to(device)
            labels = labels.to(device)
            real_targets = torch.ones(batch_size, 1, device=device)
            fake_targets = torch.zeros(batch_size, 1, device=device)

            # ---------------- تدريب D ----------------
            z = torch.randn(batch_size, args.latent_dim, device=device)
            fake_labels = torch.randint(0, args.num_classes, (batch_size,), device=device)
            fake = G(z, fake_labels)

            d_loss_real = criterion(D(real, labels), real_targets)
            d_loss_fake = criterion(D(fake.detach(), fake_labels), fake_targets)
            d_loss = d_loss_real + d_loss_fake

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # ---------------- تدريب G ----------------
            g_loss = criterion(D(fake, fake_labels), real_targets)

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            history["Generator"].append(g_loss.item())
            history["Discriminator"].append(d_loss.item())
            loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise, fixed_labels).view(-1, 1, 28, 28).cpu()
        save_sample_grid(samples, f"{args.output_dir}/samples/epoch_{epoch:03d}.png", nrow=8)
        G.train()

    plot_losses(history, f"{args.output_dir}/loss_curve.png", title="Conditional GAN Losses")
    torch.save(G.state_dict(), f"{args.output_dir}/generator_final.pth")
    torch.save(D.state_dict(), f"{args.output_dir}/discriminator_final.pth")
    print("تم التدريب وحفظ الموديلات بنجاح.")


if __name__ == "__main__":
    main()
