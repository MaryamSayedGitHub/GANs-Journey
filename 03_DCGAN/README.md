# 03 — DCGAN (Deep Convolutional GAN)

![DCGAN Architecture](../docs/images/dcgan_architecture.svg)

## الفكرة

ورقة Radford et al. (2015) جابت مجموعة قواعد معمارية خلت تدريب الـ GANs أكتر استقرارًا
وحسّنت جودة الصور بشكل كبير، عن طريق استبدال الـ Fully Connected layers بـ **Convolutional layers**.

## أهم القواعد المتبعة من الورقة

| القاعدة | ليه؟ |
|---|---|
| استبدال أي Pooling بـ Strided Convolutions (D) و Transposed Conv (G) | يخلي الشبكة تتعلم الـ downsampling/upsampling بنفسها بدل عملية ثابتة |
| استخدام BatchNorm في G و D | يثبّت التدريب ويمنع الـ mode collapse جزئيًا |
| إزالة الـ FC layers المخفية | يحافظ على الـ spatial structure بتاع الصورة |
| ReLU في G (ماعدا Tanh بالآخر)، LeakyReLU في D | تجارب عملية أثبتت إنها بتدي نتائج أفضل من Sigmoid/Tanh في النص |

## المعمارية هنا

- **Generator**: بيبدأ من `latent_dim x 1 x 1` وبيعمل 5 طبقات `ConvTranspose2d`
  لحد ما يوصل لصورة `channels x 64 x 64`.
- **Discriminator**: عكس الـ Generator تمامًا — 5 طبقات `Conv2d` بـ stride=2
  لحد ما توصل لـ score واحد `1 x 1 x 1`.

كل الأوزان بتتهيأ بـ `Normal(0, 0.02)` زي ما موصوف بالظبط في الورقة (شوف
`common/utils.py -> weights_init_normal`).

## هيكل الملفات

| ملف | الوظيفة |
|---|---|
| `model.py` | تعريف `Generator` و `Discriminator` الـ Convolutional |
| `train.py` | تدريب على MNIST بعد resize لـ 64x64 |

## تشغيل

```bash
python train.py --epochs 30 --batch-size 128 --latent-dim 100
```

## ملاحظات

- MNIST صور رمادية (channel واحد) فبنستخدم `channels=1`. لو حابب تجرب على
  CIFAR-10 أو صور ملونة، غيّر `channels=3`.
- استخدام DCGAN على MNIST هيدي نتائج ممتازة نسبيًا بسرعة لأن الـ dataset بسيط،
  لكن الفايدة الحقيقية للمعمارية دي بتظهر أكتر على datasets أعقد زي الوجوه أو الصور الطبيعية.
