"""
05_WGAN/model.py
------------------
Wasserstein GAN (Arjovsky et al. 2017)

الفروق الأساسية عن Vanilla GAN:
1. الـ Discriminator بقى اسمه "Critic" وبيرجع "score" حر (مفيش Sigmoid في الآخر).
2. الـ Loss بقى Wasserstein distance تقريبًا: E[D(real)] - E[D(fake)]
3. بعد كل update للـ critic بنعمل "weight clipping" لقيم صغيرة [-c, c]
   عشان نضمن إن الدالة تحقق شرط الـ 1-Lipschitz (مطلوب رياضيًا لصحة الـ Wasserstein distance).
4. بندرب الـ critic عدد مرات أكبر (n_critic) قبل كل خطوة تدريب للـ Generator.
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
    """
    ملاحظة: مفيش Sigmoid في الآخر، ومفيش BatchNorm (تفضيلي - بعض التطبيقات بتستخدمه
    لكن الورقة الأصلية بتحذر إنه ممكن يأثر على تقدير الـ Lipschitz constraint مع weight clipping).
    """

    def __init__(self, img_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),  # score حر، من غير Sigmoid
        )

    def forward(self, x):
        return self.net(x)
