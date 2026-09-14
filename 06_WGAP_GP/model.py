"""
06_WGAP_GP/model.py
---------------------
WGAN-GP: Improved Training of Wasserstein GANs (Gulrajani et al. 2017)

بدل Weight Clipping (اللي بيسبب مشاكل زي vanishing/exploding gradients)،
بنفرض الـ Lipschitz constraint عن طريق "Gradient Penalty":
نعاقب الموديل لو الـ gradient norm بتاع الـ critic بعيد عن 1.

ملحوظة معمارية: مفيش BatchNorm في الـ Critic هنا (بنستخدم LayerNorm بدلها لو حبينا)
لأن BatchNorm بيربط العينات ببعض داخل الـ batch وده بيبوّظ حساب الـ gradient penalty
لكل عينة على حدة. هنا استخدمنا معمارية بسيطة من غير Normalization في الـ critic.
"""

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, img_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, img_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),  # score حر بدون Sigmoid، وبدون BatchNorm
        )

    def forward(self, x):
        return self.net(x)
