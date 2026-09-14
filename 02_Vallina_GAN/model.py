"""
02_Vallina_GAN/model.py
------------------------
أبسط شكل لـ GAN زي ما اتقدّم في ورقة Goodfellow et al. 2014.
كل من الـ Generator والـ Discriminator عبارة عن Fully Connected networks بسيطة.
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
            nn.Tanh(),  # المخرجات في [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, img_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # احتمالية إن الصورة حقيقية
        )

    def forward(self, x):
        return self.net(x)
