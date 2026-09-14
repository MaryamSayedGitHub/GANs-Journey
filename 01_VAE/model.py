"""
01_VAE/model.py
----------------
Variational Autoencoder بسيط باستخدام Fully Connected layers، متدرب على MNIST.

الفكرة:
- الـ Encoder بيحوّل الصورة x لتوزيع احتمالي q(z|x) ممثّل بـ (mu, log_var).
- بنعمل "reparameterization trick" عشان نقدر نعمل backpropagation عبر عملية العينة العشوائية:
      z = mu + sigma * epsilon ,   epsilon ~ N(0, I)
- الـ Decoder بياخد z ويحاول يعيد بناء الصورة الأصلية x_hat = p(x|z).
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 20, hidden_dim: int = 400, output_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),  # يطبع القيم لـ [-1, 1] زي الـ input المطبّع
        )

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, log_var):
        """z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    @torch.no_grad()
    def sample(self, num_samples: int, device):
        """توليد صور جديدة من نويز عشوائي في الـ latent space مباشرة (بدون encoder)."""
        z = torch.randn(num_samples, self.decoder.net[0].in_features, device=device)
        return self.decoder(z)


def vae_loss_function(x_hat, x, mu, log_var):
    """
    ELBO Loss = Reconstruction Loss + KL Divergence
    - Reconstruction: بنستخدم MSE لأن الصور متطبّعة لـ [-1, 1] (Tanh output).
    - KL Divergence: بيقيس المسافة بين q(z|x) والـ prior N(0, I).
    """
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_div, recon_loss, kl_div
