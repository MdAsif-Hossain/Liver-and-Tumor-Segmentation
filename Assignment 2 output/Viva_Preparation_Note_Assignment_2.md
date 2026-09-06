# 🎓 Detailed Viva Preparation Note: Assignment 2 (Part B)
**Self-Supervised Learning (SSL) for Semantic Segmentation (Liver and Tumor)**

**Course:** CSE 438 — Digital Image Processing · East West University (Summer 2026)
**Group 01:** 
- Md. Asif Hossain (2022-3-60-007)
- Nabil Subhan (2022-3-60-063)
- K M Nudar (2022-3-60-234)

এই বিস্তারিত স্টাডি গাইডটি অ্যাসাইনমেন্ট ২ (Part B) এর ৫টি নোটবুক থেকে তৈরি করা হয়েছে। ভাইভাতে টেকনিক্যাল প্রশ্নগুলোর উত্তর দেওয়ার জন্য যা যা জানা প্রয়োজন, তার সবকিছু এখানে ধাপে ধাপে এক্সপ্লেইন করা হলো।

---

## 📌 ১. মূল উদ্দেশ্য (Core Concept of Part B)
Part A-তে আমরা **Supervised Learning** করেছিলাম, যেখানে ৯ ০৯টি লেবেলড ইমেজ (১০০%) ব্যবহার করে DeepLabV3-ResNet50 মডেল ট্রেইন করা হয়েছিল। 
কিন্তু Part B-এর মূল উদ্দেশ্য হলো **Self-Supervised Learning (SSL)** এর মাধ্যমে **Label Efficiency** অর্জন করা। 

**SSL এর কাজের ধাপ:**
1. **Pre-training (Task B):** মডেলকে প্রথমে প্রচুর আনলেবেলড ডেটা দিয়ে একা একা ফিচার বুঝতে দেওয়া হয় (৫০ ইপোক)।
2. **Transfer:** প্রি-ট্রেইনড এনকোডারকে সেগমেন্টেশন আর্কিটেকচারে (DeepLabV3/ASPP) বসানো হয়।
3. **Fine-Tuning (Task D):** এরপর খুব সামান্য লেবেলড ডেটা (১৯৫টি ইমেজ বা ২১% ডেটা) দিয়ে মডেলকে ফাইন-টিউন করা হয় (৫০ ইপোক)।

---

## 📓 ২. Notebook 1: Data Preparation (Task A)

এই নোটবুকটি ডেটাসেট তৈরি করে। Part A এর লিকেজ-সেফ স্প্লিটটিকে হুবহু ব্যবহার করা হয়েছে কোনো রিশাফলিং ছাড়াই।

### মূল কাজসমূহ:
1. **Role Mapping (ডেটাসেট স্প্লিট):**
   - **unlabelled_pretrain (১৩,৪৪৭ ইমেজেস):** Train স্প্লিটটিকে SSL প্রি-ট্রেনিংয়ের জন্য ব্যবহার করা হয়। এখানে মাস্ক (লেবেল) ফেলে দেওয়া হয়, শুধু ইমেজ ব্যবহার করা হয়।
   - **labelled_finetune (২,৫৫৩ ইমেজেস / ১৯৫ ভলিউম):** Validation স্প্লিটটিকে DeepLabV3 ফাইন-টিউনিংয়ের জন্য ব্যবহার করা হয়।
   - **eval_monitor_test (৩,১৫৮ ইমেজেস):** Test স্প্লিটটিকে ফাইনাল রেজাল্ট দেখার জন্য এবং বেস্ট চেকপয়েন্ট সিলেক্ট করার জন্য ব্যবহার করা হয় (Double Role)।

2. **2.5D Input Format:** 
   মেডিকেল CT স্ক্যান মূলত 3D ভলিউম। মডেলকে 3D কন্টেক্সট দেওয়ার জন্য একটি স্লাইসের সাথে তার আগের ও পরের স্লাইস জোড়া দিয়ে RGB চ্যানেলে [i-1, i, i+1] ইনপুট দেওয়া হয়। এতে 2D CNN (ResNet) দিয়ে 3D এর মত পারফরম্যান্স পাওয়া যায়, কিন্তু কম্পিউটেশন খরচ কম হয়।

3. **Data Leakage & Hashing:**
   - **Leakage Check:** Python এর set intersection ব্যবহার করে নিশ্চিত করা হয় যে একজন রোগীর স্লাইস যেন একই সাথে ট্রেইন এবং টেস্ট সেটে না যায়।
   - **Hashing:** hashlib.sha256 ব্যবহার করে প্রতিটি রোলের একটি ইউনিক 'Fingerprint' (hexdigest) তৈরি করা হয়, যা প্রমাণ করে যে ডেটা ম্যানিপুলেট করা হয়নি।

4. **Augmentation Pipelines:**
   - **SimCLR/BYOL:** Heavy augmentation — RandomResizedCrop, ColorJitter, GaussianBlur, HorizontalFlip.
   - **MAE:** Minimal augmentation — শুধু Crop আর Flip. (৭৫% মাস্কিং-ই এখানে মূল অগমেন্টেশন)।
   - **Fine-tuning:** ElasticTransform, GridDistortion — লিভার বা টিউমারের রিয়েল-লাইফ শেপ পরিবর্তন সিমুলেট করার জন্য।

---

## 📓 ৩. Notebook 2: SimCLR (Contrastive Learning)

SimCLR (Simple Contrastive Learning of Visual Representations) ইমেজের পজিটিভ পেয়ারকে কাছাকাছি আনে এবং নেগেটিভ পেয়ারকে দূরে ঠেলে দেয়।

### টেকনিক্যাল ওয়ার্কফ্লো:
- **Encoder:** ImageNet-init ResNet-50.
- **Projection Head:** ResNet এর শেষে একটি ছোট নেটওয়ার্ক Linear → ReLU → Linear (output dim=128) বসানো হয়। এটি শুধু প্রি-ট্রেনিংয়ের সময় ব্যবহার হয়।
- **Loss Function (NT-Xent):** 
  একই ইমেজের ২টি আলাদা augmented ভিউকে (positive pair) কাছাকাছি আনে এবং ব্যাচের অন্য সব ইমেজের ভিউগুলোকে (negative pair) দূরে ঠেলে দেয়।
- **Transfer:** প্রি-ট্রেনিং শেষে Projection Head ফেলে দেওয়া হয় এবং ResNet-50 কে DeepLabV3 এর ব্যাকবোন হিসেবে বসিয়ে ফাইন-টিউন করা হয়।

> [!IMPORTANT]
> **ভাইভা প্রশ্ন:** SimCLR-এ Projection Head কেন লাগে এবং ব্যাচ সাইজ বড় কেন হতে হয়?
> **উত্তর:** সরাসরি ফিচার থেকে লস ক্যালকুলেট করার চেয়ে ডাইমেনশন কমিয়ে (128) Projection Head দিয়ে লস ক্যালকুলেট করলে মডেল ভালো Representation শিখতে পারে। আর NT-Xent লস কাজ করতে হলে প্রচুর negative sample দরকার, তাই ব্যাচ সাইজ বড় হওয়া জরুরি।

---

## 📓 ৪. Notebook 3: BYOL (Bootstrap Your Own Latent)

BYOL হলো SimCLR-এর একটি উন্নত রূপ, যার প্রধান বৈশিষ্ট্য হলো এটি **Negative Samples** ছাড়াই কাজ করতে পারে।

### টেকনিক্যাল ওয়ার্কফ্লো:
- **Online ও Target Network:** 
  BYOL এ একই সাথে দুটি নেটওয়ার্ক থাকে। 
  - **Online Network:** গ্রেডিয়েন্ট ডিসেন্ট দিয়ে রেগুলার আপডেট হয়। এর শেষে একটি এক্সট্রা predictor লেয়ার থাকে।
  - **Target Network:** গ্রেডিয়েন্ট দিয়ে আপডেট হয় না। বরং Online Network এর weights গুলোকে একটু একটু করে **EMA (Exponential Moving Average)** দিয়ে এর ভেতরে কপি করা হয় (momentum = 0.996)।
- **Model Collapse ঠেকানো:** 
  Negative samples ছাড়া মডেলের কোলাপ্স হওয়ার কথা। কিন্তু Target Network ধীরে ধীরে আপডেট হওয়ায় (Asymmetry) মডেল কোলাপ্স হয় না।

---

## 📓 ৫. Notebook 4: MAE (Masked Autoencoder)

MAE এর কাজের ধরন Contrastive Learning থেকে সম্পূর্ণ আলাদা। এটি একটি **Reconstructive** মেথড।

### টেকনিক্যাল ওয়ার্কফ্লো:
- **Masking:** ইমেজকে ছোট ছোট প্যাচে ভাগ করে এর **৭৫%** প্যাচ র‍্যান্ডমলি মুছে ফেলা হয়।
- **Reconstruction:** এনকোডারের কাজ হলো বাকি ২৫% প্যাচ দেখে পুরো ছবির অরিজিনাল পিক্সেলগুলো রিকনস্ট্রাক্ট করা।
- **Loss:** Mean Squared Error (MSE).

> [!NOTE]
> **ভাইভা পয়েন্ট:** MAE-তে Color Jitter বা GaussianBlur এর মতো heavy augmentation ব্যবহার করা হয় না। কারণ MAE এর মূল কাজ অরিজিনাল পিক্সেল রিকনস্ট্রাক্ট করা। আগে থেকে কালার চেঞ্জ করলে মডেল ফেক পিক্সেল রিকনস্ট্রাক্ট করবে, যা উদ্দেশ্য নষ্ট করে। পিক্সেল-লেভেলে কাজ করায় এটি লিভারের ভেতরের সূক্ষ্ম টেক্সচার খুব ভালো শেখে।

---

## 📓 ৬. Notebook 5: DINOv2 (ViT-S/14 & ASPP Head)

সিরিজের সবচেয়ে অ্যাডভান্সড মডেল, যেখানে ResNet এর বদলে **ViT (Vision Transformer)** ব্যবহার করা হয়েছে।

### টেকনিক্যাল ওয়ার্কফ্লো:
- **Architecture (ViT-S/14):** Vision Transformer Small, যা ইমেজকে 14x14 pixel এর প্যাচে ভাগ করে। 
- **Resolution (224x224):** 256 সাইজ নিলে 256/14 ভগ্নাংশ হয়, তাই ইমেজের সাইজ 224x224 নেওয়া হয়েছে (224/14 = 16)।
- **Multi-Crop Augmentation:** 
  টিচার মডেলকে ইমেজের বড় অংশ (Global crop, 224x224) এবং স্টুডেন্টকে ছোট অংশ (Local crop, 98x98) দিয়ে টিচারকে কপি করতে শেখানো হয় (Self-distillation)।
- **Token → Grid Adapter:** 
  ViT এর আউটপুট হলো 1D token sequence (256 tokens)। কিন্তু DeepLabV3 (ASPP) হেড কাজ করে 2D spatial grid-এ। তাই 	oken_map ফাংশন দিয়ে 256 tokens কে sqrt(256)=16 করে 16x16 এর 2D grid-এ reshape করা হয়।
- **Frozen vs Fine-Tuned:** 
  অ্যাসাইনমেন্টের রিকোয়ারমেন্ট অনুযায়ী এনকোডারকে লক (Frozen / Linear-probe) করে এবং আনলক করে এক্সপেরিমেন্ট করা হয়েছে। আনলক (Fine-tuned) অবস্থায় রেজাল্ট অনেক ভালো আসে।

---

## 📊 ৭. Final Comparison & Error Analysis (Task E & F)

সবগুলো মডেলের রেজাল্ট কম্পেয়ার করে part-b-nb5-final-comparison-label-efficiency.ipynb এ কনক্লুশন টানা হয়েছে।

### Label Efficiency Caveats (রিপোর্টে যা লেখা হয়েছে):
1. **Label Budget:** Part A তে ৯০৯টি লেবেলড ইমেজ (১০০%) ব্যবহার করা হয়েছিল। কিন্তু Part B তে মাত্র ১৯৫টি (২১%) লেবেলড ইমেজ ব্যবহার করা হয়েছে।
2. **Input Resolution Gap:** Part A তে রেজোলিউশন ছিল 512x512, আর Part B তে ViT এর বাধ্যবাধকতার কারণে 224x224। ছোট রেজোলিউশনে পাতলা টিউমার ডিটেক্ট করা অনেক কঠিন।
3. **Double Role of Test Split:** Test স্প্লিটকে চেকপয়েন্ট সিলেকশন এবং ফাইনাল মেট্রিক্স দুটোর জন্যই ব্যবহার করায় রেজাল্ট সামান্য অপটিমিস্টিক হতে পারে।

### Error Analysis (Task F - Failure Mode):
- **Observation:** সবচেয়ে খারাপ পারফর্ম করা স্লাইসগুলোতে টিউমারগুলো খুবই ছোট (tiny) অথবা বাউন্ডারির একদম কাছে (boundary-adjacent lesions)।
- **Error Map:** এরর ম্যাপগুলো বেশিরভাগই **লাল (Tumour false-negatives)**, সায়ান (Cyan) নয়। অর্থাৎ, মডেল খুব *কনজারভেটিভ* — সে ভুল করে অন্য কিছুকে টিউমার বলার চেয়ে, আসল টিউমারকেই মিস করে বেশি। 
- **Part A vs Part B:** Part A এর সুপারভাইজড মডেলেও ঠিক একই ইমেজগুলোতে এবং একই কারণে মডেল ফেইল করেছিল। এর মানে হলো সমস্যাটা SSL মেথডের না, বরং **ডেটার (Lesion size and contrast)**। 

### Best Models:
এত প্রতিকূলতা (২১% ডেটা, ছোট রেজোলিউশন) সত্ত্বেও **DINOv2 (mIoU: 0.714)** এবং **MAE (mIoU: 0.691)** সুপারভাইজড বেসলাইনের প্রায় **৮৩-৮৬%** পারফরম্যান্স অর্জন করতে পেরেছে! 

---
*Good luck with the final presentation and Viva!*
