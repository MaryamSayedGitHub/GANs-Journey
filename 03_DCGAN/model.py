"""
03_DCGAN/model.py
-------------------
Deep Convolutional GAN (Radford et al. 2015)

أهم القواعد المتبعة من الورقة الأصلية:
- استبدال أي Pooling بـ Strided Convolutions (في D) و Transposed Convolutions (في G).
- استخدام BatchNorm في كل من G و D (ماعدا الطبقة الأخيرة في كل منهم).
- إزالة الـ Fully Connected layers المخفية.
- ReLU في G لكل الطبقات ماعدا الأخيرة (Tanh)، LeakyReLU في D لكل الطبقات.
"""

import torch.nn as nn


class Generator(nn.Module):
    """
    مدخل: latent vector z بشكل (N, latent_dim, 1, 1)
    مخرج: صورة (N, channels, 64, 64)
    """

    def __init__(self, latent_dim: int = 100, feature_maps: int = 64, channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            # (latent_dim, 1, 1) -> (fm*8, 4, 4)
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),

            # -> (fm*4, 8, 8)
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # -> (fm*2, 16, 16)
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # -> (fm, 32, 32)
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # -> (channels, 64, 64)
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """
    مدخل: صورة (N, channels, 64, 64)
    مخرج: احتمالية (N, 1, 1, 1) -> real/fake
    """

    def __init__(self, feature_maps: int = 64, channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            # (channels, 64, 64) -> (fm, 32, 32)
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (fm*2, 16, 16)
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (fm*4, 8, 8)
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (fm*8, 4, 4)
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (1, 1, 1)
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)
