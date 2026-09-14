# 05 — WGAN (Wasserstein GAN)

![WGAN Architecture](../docs/images/wgan_architecture.svg)

## المشكلة اللي بيحلها

في الـ GAN العادي، دالة الخسارة مبنية على **Jensen-Shannon Divergence** ضمنيًا،
واللي بتعاني من مشكلة إن الـ gradient بيبقى صفر أو شبه صفر لما التوزيعين (الحقيقي والمولّد)
ميكونوش متداخلين (وده شائع جدًا في بداية التدريب) → **Vanishing Gradients** وتدريب غير مستقر.

## الحل: Wasserstein Distance

ورقة Arjovsky et al. (2017) اقترحت استخدام **Earth Mover's Distance (Wasserstein-1)**
بدل JS Divergence، لأنها بتوفر gradient أفضل حتى لو التوزيعين بعيدين عن بعض تمامًا.

```
W(P_real, P_fake) ≈ max_{||D||_L <= 1}  E[D(real)] - E[D(fake)]
```

## أهم 3 تغييرات عن Vanilla GAN

| # | التغيير | السبب |
|---|---|---|
| 1 | الـ Discriminator بقى اسمه **Critic** وبيرجع score حر (من غير Sigmoid) | لأننا بنقيس مسافة مش احتمالية |
| 2 | **Weight Clipping**: `w = clip(w, -c, c)` بعد كل update | عشان نفرض شرط الـ **1-Lipschitz** المطلوب رياضيًا لصحة تقدير الـ Wasserstein distance |
| 3 | تدريب الـ Critic **n_critic** مرة (افتراضيًا 5) قبل كل خطوة تدريب لـ G | عشان الـ Critic يوصل لتقدير كويس للمسافة قبل ما نحدّث G بناءً عليه |

الـ Loss بقت:
```
Critic Loss    = -( E[D(real)] - E[D(fake)] )
Generator Loss = -E[D(fake)]
```

## هيكل الملفات

| ملف | الوظيفة |
|---|---|
| `model.py` | تعريف `Generator` و `Critic` (بدون Sigmoid في الآخر) |
| `train.py` | حلقة تدريب مع Weight Clipping و n_critic، باستخدام RMSProp |

## تشغيل

```bash
python train.py --epochs 50 --n-critic 5 --clip-value 0.01
```

## عيوب Weight Clipping (وليه محتاجين WGAN-GP بعد كده)

- لو `clip_value` صغير جدًا → الـ gradients بتموت (vanishing).
- لو `clip_value` كبير جدًا → الـ gradients بتنفجر (exploding)، والتدريب بياخد وقت
  طويل عشان الأوزان توصل لحدود الـ clipping.
- الـ Critic بيميل يتعلم دوال بسيطة جدًا (شبه خطية) عشان يفضل جوه حدود الـ clipping،
  وده بيقلل من قدرته التعبيرية (capacity).

الحل لكل المشاكل دي هنلاقيه في **WGAN-GP** في الفولدر الجاي.
