# 02 — Vanilla GAN

![GAN Architecture](../docs/images/gan_architecture.svg)

## الفكرة

أول ورقة GAN اتقدمت من Ian Goodfellow سنة 2014. الفكرة عبارة عن لعبة
**Minimax** بين شبكتين:

- **Generator (G)**: بياخد نويز عشوائي `z` ويحاول يحوّله لصورة تشبه الـ dataset.
- **Discriminator (D)**: بياخد صورة (حقيقية أو مولّدة) ويحاول يحدد هل هي حقيقية ولا لأ.

```
min_G max_D   E_x~p_data[log D(x)]  +  E_z~p_z[log(1 - D(G(z)))]
```

الاتنين بيتدربوا في نفس الوقت في لعبة تنافسية: الـ Discriminator بيتعلم يميز أحسن،
والـ Generator بيتعلم يخدع الـ Discriminator أحسن، لحد ما (نظريًا) الـ Generator
يوصل لتوزيع بيانات مطابق تقريبًا للتوزيع الحقيقي.

## المعمارية هنا

كل من G و D عبارة عن **Fully Connected (Linear) layers** بسيطة (مفيش Convolutions).
ده بيخليها سهلة الفهم كخطوة أولى، لكنه بيحد من جودة الصور المولّدة مقارنة بـ DCGAN.

- **Generator**: `Linear -> LeakyReLU -> BatchNorm` (تصاعديًا) لحد `Tanh` في الآخر.
- **Discriminator**: `Linear -> LeakyReLU` لحد `Sigmoid` في الآخر (احتمالية real/fake).

## هيكل الملفات

| ملف | الوظيفة |
|---|---|
| `model.py` | تعريف `Generator` و `Discriminator` |
| `train.py` | حلقة تدريب الـ minimax game (تحديث D ثم G بالتبادل) |

## تشغيل

```bash
python train.py --epochs 50 --batch-size 128 --latent-dim 100
```

## مشاكل معروفة في Vanilla GAN (وليه محتاجين DCGAN وWGAN بعد كده)

1. **عدم استقرار التدريب (Training Instability)**: توازن G و D صعب جدًا، ولو
   D قوي أوي بيبقى الـ gradient اللي بيوصل لـ G ضعيف جدًا (**Vanishing Gradients**).
2. **Mode Collapse**: الـ Generator ممكن "يغش" ويولّد نفس الصورة (أو مجموعة صغيرة
   من الصور) بدل ما يغطي كل تنوع الـ dataset.
3. الاستخدام المباشر لـ FC layers بيضيع الـ spatial structure بتاع الصور،
   عشان كده DCGAN جه بعد كده يستخدم Convolutions بدلها.
