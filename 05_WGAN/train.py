"""
05_WGAN/train.py
------------------
تدريب WGAN على MNIST باستخدام RMSProp (زي الورقة الأصلية) و Weight Clipping.

تشغيل:
    python train.py --epochs 50 --n-critic 5 --clip-value 0.01
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.utils import get_device, get_mnist_dataloader, save_sample_grid, plot_losses  # noqa: E402
from model import Generator, Critic  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--n-critic", type=int, default=5, help="عدد مرات تدريب الـ critic لكل خطوة G")
    parser.add_argument("--clip-value", type=float, default=0.01, help="حد الـ weight clipping")
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    dataloader = get_mnist_dataloader(batch_size=args.batch_size)

    G = Generator(latent_dim=args.latent_dim).to(device)
    C = Critic().to(device)

    # الورقة الأصلية بتنصح بـ RMSProp بدل Adam مع WGAN
    opt_G = torch.optim.RMSprop(G.parameters(), lr=args.lr)
    opt_C = torch.optim.RMSprop(C.parameters(), lr=args.lr)

    history = {"Generator": [], "Critic": []}
    fixed_noise = torch.randn(64, args.latent_dim, device=device)

    step = 0
    for epoch in range(1, args.epochs + 1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, _ in loop:
            batch_size = images.size(0)
            real = images.view(batch_size, -1).to(device)

            # ---------------- تدريب الـ Critic ----------------
            z = torch.randn(batch_size, args.latent_dim, device=device)
            fake = G(z).detach()

            c_loss = -(torch.mean(C(real)) - torch.mean(C(fake)))

            opt_C.zero_grad()
            c_loss.backward()
            opt_C.step()

            # Weight Clipping: أهم خطوة في WGAN عشان نحقق الـ Lipschitz constraint
            for p in C.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)

            step += 1

            # ---------------- تدريب الـ Generator كل n_critic خطوة ----------------
            if step % args.n_critic == 0:
                z = torch.randn(batch_size, args.latent_dim, device=device)
                fake = G(z)
                g_loss = -torch.mean(C(fake))

                opt_G.zero_grad()
                g_loss.backward()
                opt_G.step()

                history["Generator"].append(g_loss.item())
                history["Critic"].append(c_loss.item())
                loop.set_postfix(c_loss=c_loss.item(), g_loss=g_loss.item())

        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise).view(-1, 1, 28, 28).cpu()
        save_sample_grid(samples, f"{args.output_dir}/samples/epoch_{epoch:03d}.png")
        G.train()

    plot_losses(history, f"{args.output_dir}/loss_curve.png", title="WGAN Losses")
    torch.save(G.state_dict(), f"{args.output_dir}/generator_final.pth")
    torch.save(C.state_dict(), f"{args.output_dir}/critic_final.pth")
    print("تم التدريب وحفظ الموديلات بنجاح.")


if __name__ == "__main__":
    main()
