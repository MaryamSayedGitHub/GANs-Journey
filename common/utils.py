"""
common/utils.py
----------------
دوال مشتركة بيستخدمها كل الموديلات في الريبو (VAE, GAN, DCGAN, cGAN, WGAN, WGAN-GP)
عشان نتجنب تكرار نفس الكود في كل فولدر.
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt


def get_device() -> torch.device:
    """يرجع GPU لو متاح، غير كده CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mnist_dataloader(batch_size: int = 128, image_size: int = 28, root: str = "./data"):
    """
    بيحمل MNIST (بيتنزل تلقائيًا أول مرة) ويرجع DataLoader جاهز.
    مستخدم في VAE / Vanilla GAN / DCGAN / cGAN.
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # يطبع القيم لـ [-1, 1]
    ])
    dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def weights_init_normal(m: nn.Module):
    """
    تهيئة الأوزان بطريقة DCGAN paper: Normal(0, 0.02) للـ Conv/BatchNorm layers.
    بيتحط عن طريق model.apply(weights_init_normal)
    """
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_sample_grid(tensor_images: torch.Tensor, path: str, nrow: int = 8, title: str = None):
    """
    بيحفظ شبكة من الصور المولّدة كملف PNG. الصور المفروض تكون بالفعل في range [-1,1] أو [0,1].
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = make_grid(tensor_images, nrow=nrow, normalize=True)
    save_image(grid, path)
    if title:
        print(f"[saved] {title} -> {path}")


def plot_losses(loss_dict: dict, path: str, title: str = "Training Losses"):
    """
    بيرسم منحنى الـ losses عبر الـ epochs ويحفظه كصورة.
    loss_dict: {"Generator": [...], "Discriminator": [...]}
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    for name, values in loss_dict.items():
        plt.plot(values, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(path)
    plt.close()


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """يرجع الصور من [-1, 1] لـ [0, 1] عشان العرض."""
    return (tensor + 1) / 2


def count_parameters(model: nn.Module) -> int:
    """عدد الـ trainable parameters في الموديل - مفيد للمقارنة بين الموديلات."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
