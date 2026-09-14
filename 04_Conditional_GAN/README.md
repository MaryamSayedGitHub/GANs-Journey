# 04 — Conditional GAN (cGAN)

![Conditional GAN Architecture](../docs/images/cgan_architecture.svg)

## الفكرة

في الـ Vanilla GAN، مفيش طريقة تتحكم بيها في نوع الصورة اللي هتتولد — الـ Generator
بياخد نويز عشوائي بس ومش عارف نولّد رقم معين بالتحديد.

ورقة Mirza & Osindero (2014) حلّت المشكلة دي بإضافة **معلومة إضافية (condition) y**
لكل من G و D، غالبًا بتكون الـ class label.

```
G(z, y) -> صورة من الفئة y
D(x, y) -> هل الصورة x حقيقية بالنسبة للفئة y؟
```

## إزاي بنعمل كده تقنيًا؟

1. بنحوّل الـ label (رقم صحيح من 0 لـ 9) لـ **embedding vector** باستخدام `nn.Embedding`.
2. بنعمل **concatenate** للـ embedding مع الـ input الأصلي:
   - في G: `concat(z, embed(y))`
   - في D: `concat(image, embed(y))`
3. الباقي زي الـ GAN العادي — نفس الـ minimax game، لكن الاتنين "عارفين" الفئة y.

## المعمارية هنا

FC layers زي Vanilla GAN، لكن مع طبقة `nn.Embedding(num_classes, embed_dim)`
إضافية في كل من G و D.

## هيكل الملفات

| ملف | الوظيفة |
|---|---|
| `model.py` | تعريف `Generator` و `Discriminator` مع دعم الـ label conditioning |
| `train.py` | تدريب + توليد عينات ثابتة لكل رقم من 0 لـ 9 لمتابعة التقدم |

## تشغيل

```bash
python train.py --epochs 50 --batch-size 128 --num-classes 10
```

## توليد رقم محدد بعد التدريب

```python
import torch
from model import Generator

G = Generator()
G.load_state_dict(torch.load("outputs/generator_final.pth"))
G.eval()

z = torch.randn(1, 100)
label = torch.tensor([7])   # عايزين نولّد رقم 7 تحديدًا
with torch.no_grad():
    image = G(z, label).view(28, 28)
```

## ملاحظات

- كل ما الـ embedding dimension أكبر، الموديل بيقدر يميز بين الفئات أحسن، لكن
  ده بيزوّد عدد الـ parameters. `embed_dim=50` كفاية لـ 10 فئات بسيطة زي MNIST.
- الفكرة دي أساس لمعماريات أعقد كتير زي **Pix2Pix** و **StyleGAN** اللي بتستخدم
  conditioning أكتر تعقيدًا (صورة كاملة بدل label بسيط).
