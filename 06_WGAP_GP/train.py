"""
06_WGAP_GP/train.py
---------------------
تدريب WGAN-GP على MNIST باستخدام Adam (بخلاف WGAN العادي) و Gradient Penalty.

تشغيل:
    python train.py --epochs 50 --n-critic 5 --lambda-gp 10
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--lambda-gp", type=float, default=10.0, help="معامل الـ gradient penalty")
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def compute_gradient_penalty(critic, real, fake, device):
    """
    بنحسب الـ gradient penalty على نقاط عشوائية بين real و fake:
        x_hat = eps * real + (1 - eps) * fake
        GP = E[(||grad_x_hat D(x_hat)||_2 - 1)^2]
    """
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, device=device).expand_as(real)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)

    scores = critic(x_hat)
    grad_outputs = torch.ones_like(scores, device=device)

    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=x_hat,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    dataloader = get_mnist_dataloader(batch_size=args.batch_size)

    G = Generator(latent_dim=args.latent_dim).to(device)
    C = Critic().to(device)

    # مع Gradient Penalty، Adam بيشتغل كويس (بخلاف WGAN العادي مع Weight Clipping)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=args.lr, betas=(0.5, 0.9))

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

            gp = compute_gradient_penalty(C, real, fake, device)
            c_loss = -(torch.mean(C(real)) - torch.mean(C(fake))) + args.lambda_gp * gp

            opt_C.zero_grad()
            c_loss.backward()
            opt_C.step()

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
                loop.set_postfix(c_loss=c_loss.item(), g_loss=g_loss.item(), gp=gp.item())

        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise).view(-1, 1, 28, 28).cpu()
        save_sample_grid(samples, f"{args.output_dir}/samples/epoch_{epoch:03d}.png")
        G.train()

    plot_losses(history, f"{args.output_dir}/loss_curve.png", title="WGAN-GP Losses")
    torch.save(G.state_dict(), f"{args.output_dir}/generator_final.pth")
    torch.save(C.state_dict(), f"{args.output_dir}/critic_final.pth")
    print("تم التدريب وحفظ الموديلات بنجاح.")


if __name__ == "__main__":
    main()
