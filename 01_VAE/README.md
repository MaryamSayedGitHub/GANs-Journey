# 01 — Variational Autoencoder (VAE)

![VAE Architecture](../docs/images/vae_architecture.svg)

## الفكرة

الـ VAE مش موديل Adversarial زي باقي الريبو ده، لكنه بداية منطقية عشان نفهم فكرة
**Latent Space** والتوليد الاحتمالي قبل ما ندخل عالم GANs.

بدل ما الـ Autoencoder العادي يضغط الصورة لنقطة واحدة ثابتة في الـ latent space،
الـ VAE بيضغطها لـ **توزيع احتمالي** (Gaussian) ممثّل بـ:

- `mu` (المتوسط)
- `log_var` (لوغاريتم الـ variance، بنستخدمه بدل الـ variance مباشرة لضمان الاستقرار العددي)

### Reparameterization Trick

المشكلة: مينفعش نعمل backpropagation من خلال عملية "أخذ عينة عشوائية" مباشرة.
الحل:

```
z = mu + sigma * epsilon        حيث epsilon ~ N(0, I)
```

كده الـ randomness بقت في `epsilon` بس، والـ gradient بيعدي بسهولة عبر `mu` و `sigma`.

### دالة الخسارة (ELBO)

```
Loss = Reconstruction_Loss(x, x_hat) + KL_Divergence(q(z|x) || N(0, I))
```

- **Reconstruction Loss**: بيقيس مدى قرب الصورة المعاد بناؤها من الأصلية.
- **KL Divergence**: بيجبر الـ latent space إنه يكون قريب من توزيع طبيعي قياسي،
  وده اللي بيخلينا نقدر نولّد صور جديدة بعد التدريب عن طريق أخذ عينة عشوائية من `N(0, I)` مباشرة.

## هيكل الملفات

| ملف | الوظيفة |
|---|---|
| `model.py` | تعريف `Encoder`, `Decoder`, `VAE`, ودالة `vae_loss_function` |
| `train.py` | حلقة التدريب الكاملة + حفظ عينات وخسائر كل epoch |

## تشغيل

```bash
python train.py --epochs 20 --batch-size 128 --latent-dim 20
```

المخرجات هتتحفظ في `outputs/`:
- `outputs/samples/epoch_XXX.png` — عينات مولّدة بعد كل epoch
- `outputs/loss_curve.png` — منحنى الخسارة
- `outputs/vae_final.pth` — أوزان الموديل النهائية

## ملاحظات مهمة

- الصور المولّدة من VAE عادة بتكون **أكثر ضبابية (blurry)** مقارنة بالـ GANs، وده
  بسبب استخدام Reconstruction Loss (MSE) اللي بتميل تعمل "متوسط" بين الاحتمالات الممكنة.
- زيادة `latent_dim` بتحسّن جودة إعادة البناء لكن ممكن تقلل من جودة الـ interpolation
  في الـ latent space.
