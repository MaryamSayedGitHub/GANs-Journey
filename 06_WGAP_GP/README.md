# 06 — WGAN-GP (Wasserstein GAN with Gradient Penalty)

![WGAN-GP Architecture](../docs/images/wgan_gp_architecture.svg)

## المشكلة اللي بيحلها

WGAN حل مشكلة الاستقرار الأساسية في GANs، لكن طريقة فرض الـ Lipschitz constraint
بتاعته (**Weight Clipping**) نفسها كانت مصدر مشاكل جديدة (vanishing/exploding
gradients، وقدرة تعبيرية محدودة للـ critic).

## الحل: Gradient Penalty

ورقة Gulrajani et al. (2017) اقترحت بدل ما "نقص" الأوزان، **نعاقب** الموديل
مباشرة لو الـ gradient norm بتاعه بعيد عن 1، وده طريقة أنعم وأكثر مباشرة لفرض
شرط الـ 1-Lipschitz.

### خطوات حساب الـ Gradient Penalty

1. ناخد نقطة عشوائية بين صورة حقيقية وصورة مولّدة:
   ```
   x_hat = eps * x_real + (1 - eps) * x_fake ,   eps ~ Uniform(0, 1)
   ```
2. نحسب الـ gradient بتاع الـ critic بالنسبة لـ `x_hat`.
3. نعاقب أي انحراف عن norm = 1:
   ```
   GP = E[ (||∇_x_hat D(x_hat)||_2 - 1)^2 ]
   ```
4. دالة الخسارة النهائية للـ critic:
   ```
   Critic Loss = -( E[D(real)] - E[D(fake)] ) + lambda * GP
   ```
   حيث `lambda` (افتراضيًا 10) بيتحكم في قوة العقاب.

## الفروق عن WGAN العادي

| | WGAN | WGAN-GP |
|---|---|---|
| فرض Lipschitz constraint | Weight Clipping | Gradient Penalty |
| الـ Optimizer | RMSProp | Adam (β1=0.5, β2=0.9) |
| BatchNorm في الـ Critic | ممكن تستخدم | **لأ** — بتبوّظ حساب الـ gradient لكل عينة على حدة |
| استقرار التدريب | متوسط | أفضل بكتير |
| جودة الصور | جيدة | أفضل عمومًا |

## هيكل الملفات

| ملف | الوظيفة |
|---|---|
| `model.py` | تعريف `Generator` و `Critic` (بدون BatchNorm في الـ Critic) |
| `train.py` | يحتوي `compute_gradient_penalty()` بالإضافة لحلقة التدريب الكاملة |

## تشغيل

```bash
python train.py --epochs 50 --n-critic 5 --lambda-gp 10
```

## نصيحة عملية

لو لاحظت إن قيمة الـ Gradient Penalty مش بتقل مع الوقت، جرّب:
- تقليل الـ learning rate.
- التأكد إن مفيش BatchNorm في الـ critic (لازم يتشال لو موجود).
- زيادة `n_critic` قليلاً عشان الـ critic ياخد وقت أطول يستقر قبل كل تحديث لـ G.
