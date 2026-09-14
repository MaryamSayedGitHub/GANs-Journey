"""
04_Conditional_GAN/model.py
-----------------------------
Conditional GAN (Mirza & Osindero, 2014)

الفكرة: بندي كل من G و D معلومة إضافية عن الـ class label y، عن طريق
تحويل الـ label لـ embedding vector وعمل concatenate له مع الـ input.
كده نقدر نتحكم في نوع الصورة اللي بتتولد (مثلاً نولّد رقم "7" تحديدًا).
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, num_classes: int = 10, img_dim: int = 784, embed_dim: int = 50):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        label_emb = self.label_embedding(labels)          # (N, embed_dim)
        x = torch.cat([z, label_emb], dim=1)               # (N, latent_dim + embed_dim)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes: int = 10, img_dim: int = 784, embed_dim: int = 50):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(img_dim + embed_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        label_emb = self.label_embedding(labels)
        x = torch.cat([img, label_emb], dim=1)
        return self.net(x)
