# CNN — CIFAR-10 Image Classification with Transfer Learning

## Project Summary

Built and compared Convolutional Neural Networks (CNN) from scratch, with data augmentation, and using transfer learning (ResNet18) on the CIFAR-10 dataset. This project demonstrates why CNNs dominate image classification tasks and how transfer learning dramatically outperforms training from scratch.

**Dataset:** [uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10)

**Business Problem:** Can we automatically classify real world objects in images — powering applications like product cataloguing, visual search, quality control, and content moderation?

---

## Results Summary

| Model | Accuracy | Training Time |
|---|---|---|
| CNN from scratch | 82.14% | ~30 mins CPU |
| CNN + Augmentation (20 epochs) | 78.13% | ~35 mins CPU |
| ResNet18 (wrong config) | 40.26% | ~10 mins GPU |
| ResNet18 (fixed, transfer learning) | 93.07% | ~10 mins GPU |

**Transfer learning beats everything** — 93% accuracy by fine-tuning a pretrained model vs 82% training from scratch.

---

## Dataset Overview

```
Training images: 50,000
Test images:     10,000
Image size:      32×32 pixels, RGB (3 channels)
Classes:         10 — airplane, automobile, bird, cat, deer,
                      dog, frog, horse, ship, truck
```

Real world colour images — significantly harder than MNIST digits.

---

## Key Learnings

### 1. Why CNNs Beat FNNs on Images

Feedforward Neural Networks flatten images into a 1D list:
```
32×32×3 image → 3,072 numbers → loses all spatial information
Pixel at (3,4) has no relationship to pixel at (3,5)
```

CNNs keep the 2D structure:
```
32×32×3 image → stays as a grid
Nearby pixels processed together
Filters detect edges, curves, textures, shapes
```

This spatial awareness is why CNNs dominate every image task.

### 2. CNN Architecture — Three Conv Blocks

```
Input: 32×32×3 RGB image
         ↓
Conv Block 1 → 32 filters, 32×32 → 16×16 (learns edges, colours)
         ↓
Conv Block 2 → 64 filters, 16×16 → 8×8 (learns shapes, textures)
         ↓
Conv Block 3 → 128 filters, 8×8 → 4×4 (learns object parts)
         ↓
Classifier → Flatten → Linear(2048,512) → Linear(512,10)
         ↓
Output: 10 class probabilities
Total parameters: 1,342,122
```

Each block doubles the number of filters while halving spatial dimensions — standard CNN design pattern used in VGG, ResNet, and most production models.

### 3. New Layers Introduced

**Conv2d** — slides small filters across the image detecting patterns:
```python
nn.Conv2d(3, 32, kernel_size=3, padding=1)
# 3 input channels (RGB) → 32 learned filters → each 3×3 pixels
# padding=1 → output stays same spatial size as input
```

**MaxPool2d** — reduces spatial size by half, keeps strongest signal:
```python
nn.MaxPool2d(2, 2)
# Takes maximum value in each 2×2 block
# 32×32 → 16×16, 4x fewer values to process
# Adds translation invariance — cat at (10,10) same as cat at (10,12)
```

**Dropout2d** — drops entire feature map channels during training:
```python
nn.Dropout2d(0.2)  # drops 20% of channels randomly
# Different from Dropout which drops individual neurons
# Forces other filters to learn redundant representations
```

**BatchNorm2d** — normalises output of each conv layer:
```python
nn.BatchNorm2d(32)  # 32 matches number of filters
# Keeps values stable across multiple conv layers
# Critical for training deep CNNs without instability
```

### 4. Data Augmentation — When, Why, and When Not To

#### What It Is

Data augmentation artificially creates variations of training images so the model learns to recognise objects regardless of orientation, lighting, position, or slight distortion:

```python
transforms.RandomHorizontalFlip(p=0.5)  # flip 50% of images
transforms.RandomCrop(32, padding=4)     # shift object position
transforms.ColorJitter(brightness=0.2)  # vary lighting conditions
transforms.RandomRotation(15)           # tilt up to 15 degrees
```

Every epoch each image looks slightly different — the model never sees the exact same version twice. This forces it to learn the underlying object rather than memorising specific pixel arrangements.

#### Why It Works

Without augmentation the model memorises exact training images:
```
Training image: cat facing right, bright lighting, centered
Model learns:   this exact arrangement of pixels = cat

New image: cat facing left, dim lighting, off-center
Model thinks: this doesn't look like my training cats → wrong prediction
```

With augmentation the model learns the concept:
```
Training images: cat facing left, right, cropped, bright, dark, rotated
Model learns:    the underlying shape and texture that defines a cat

New image: cat in any orientation or lighting
Model thinks: these features match cat regardless of arrangement → correct
```

#### When To Use Augmentation

✅ **Small datasets** — fewer than 10,000 images per class. Augmentation multiplies effective training data without collecting more.

✅ **When real world inputs will vary** — if your product processes user uploaded photos taken in different lighting, angles, and conditions — augmentation teaches the model to handle that variability.

✅ **When model is overfitting** — large gap between training and test accuracy signals the model memorised training data. Augmentation forces generalisation.

✅ **Image classification, object detection, medical imaging** — any vision task where the object identity doesn't change with orientation or lighting.

✅ **Before labelling more data** — augmentation is free. Collecting and labelling new images is expensive. Always try augmentation first.

#### When NOT To Use Augmentation

❌ **When orientation matters** — don't use horizontal flip for digit recognition (6 flipped is 9) or text recognition (letters are orientation-specific).

❌ **When colour is diagnostic** — don't use colour jitter for medical skin lesion classification where specific colour patterns indicate specific conditions.

❌ **On test data — ever** — test data must represent real world input exactly as it arrives. Augmenting test data gives falsely optimistic results that don't reflect production performance.

❌ **When you have abundant data** — with 1 million+ images the model will naturally see enough variation. Augmentation adds compute cost with minimal benefit.

❌ **Tabular data** — augmentation is almost exclusively a computer vision technique. It doesn't apply to structured/tabular ML problems.

#### The Epoch Trap — Why Augmentation Appeared To Hurt

```
Without augmentation — epoch 20:
Training accuracy: 81.84% ← converged, learned the data
Test accuracy:     82.14%

With augmentation — epoch 20:
Training accuracy: 68.47% ← still learning, harder data
Test accuracy:     78.13%
```

Augmentation appeared to hurt because 20 epochs was not enough. Every epoch the model sees different versions of the same images — it needs more iterations to converge:

```
Without augmentation: 50,000 unique images × 20 epochs = 1,000,000 training examples
With augmentation:    50,000 images, infinite variations × 20 epochs = still learning
```

At 40+ epochs augmentation consistently outperforms no augmentation by 3-5%. The model that seemed to fail at 20 epochs would have won at 40.

**PM lesson:** Never evaluate augmentation at the same epoch count as a non-augmented baseline. You are comparing a converged model against one that is still learning.

#### The Right Way To Choose Augmentation Techniques

Ask one question for each technique: **"Does this transformation preserve the label?"**

```
Horizontal flip on a cat → still a cat ✅ use it
Vertical flip on a digit 6 → looks like 9 ❌ don't use it
Colour jitter on a car → still a car ✅ use it
Colour jitter on a traffic light → changes meaning ❌ don't use it
Rotation on an airplane → still an airplane ✅ use it
Rotation on text → changes the letter ❌ don't use it
```

This single question prevents most augmentation mistakes in production.

### 5. Transfer Learning — The Most Important Lesson

#### What It Is

Transfer learning reuses a model trained on a large dataset as the starting point for a new task. Instead of learning from scratch, you borrow knowledge already extracted from millions of images:

```
ImageNet pretrained ResNet18:
Layer 1 → already knows how to detect edges and corners
Layer 2 → already knows how to detect textures and patterns
Layer 3 → already knows how to detect shapes and object parts
Layer 4 → knows ImageNet specific high level features (needs adapting)
FC      → replaced entirely with CIFAR-10 classifier
```

ResNet18 was pretrained on ImageNet — 1.2 million images across 1,000 categories. It has already learned universal visual features that apply to almost any image task.

#### Why It Works

Visual features are hierarchical and largely universal:

```
Edge detection → works for cats, cars, x-rays, satellite images
Texture detection → works for fur, metal, fabric, tissue
Shape detection → works for wheels, faces, buildings, organs
```

A model trained on 1.2 million diverse images has seen enough variation to learn genuinely universal low and mid level features. Your CIFAR-10 task only needs to teach it the high level CIFAR-10 specific distinctions.

```
Training from scratch:
Model must learn edges AND textures AND shapes AND CIFAR classes
From only 50,000 images

Transfer learning:
Edges, textures, shapes already learned from 1.2M images
Model only needs to learn CIFAR-10 specific distinctions
From 50,000 images
```

#### Results

```
Training only 1.2% of the network (133,898 parameters)
Getting 99% of the knowledge for free
Accuracy: 93.07% vs 82.14% from scratch
11% improvement, 10x less training time
```

#### When To Use Transfer Learning

✅ **Limited data** — transfer learning works with as few as 500-1,000 images per class. Training from scratch needs 10,000+.

✅ **Similar domain** — your task involves natural images (photos of objects, animals, products, people). ImageNet pretrained models transfer well.

✅ **Time constrained** — product deadline is tight. Transfer learning reaches production accuracy in hours, not weeks.

✅ **Limited compute budget** — fine-tuning costs a fraction of training from scratch. Training ResNet18 from scratch = days on GPU. Fine-tuning = hours.

✅ **Standard vision tasks** — image classification, object detection, image segmentation. All benefit enormously from pretrained features.

✅ **New product with cold start** — launching a new feature that needs image understanding but has no historical training data yet.

✅ **Iterating rapidly** — when you need to test multiple product hypotheses quickly, transfer learning lets you validate each in hours instead of days.

#### When NOT To Use Transfer Learning

❌ **Highly specialised domains with no relevant pretrained models** — satellite imagery at specific wavelengths, microscopy of rare cell types, industrial defect detection with proprietary equipment. ImageNet features don't transfer to these domains.

❌ **When data distribution is completely different** — medical X-rays, MRI scans, thermal imaging. Pixel statistics are fundamentally different from natural photos. Pretrained features may actually hurt.

❌ **When you need full control over learned representations** — research applications where understanding exactly what the model learned matters more than accuracy.

❌ **Very large proprietary datasets** — if you have 10 million domain-specific images, training from scratch may eventually outperform fine-tuning. The pretrained model's generalisation advantage disappears with enough domain data.

❌ **When the pretrained model is too large for your infrastructure** — ResNet18 is small (11M parameters). But ResNet152 or ViT-Large may be too slow for real time mobile inference. Don't use pretrained models larger than your infrastructure can serve.

#### How Much To Unfreeze — The Fine-tuning Strategy

The decision of which layers to freeze vs unfreeze is one of the most important transfer learning decisions:

```
Strategy 1 — Freeze everything, train only classifier:
Use when: task is very similar to pretrained task, large dataset
Risk: high level features may not match your specific classes

Strategy 2 — Freeze early layers, unfreeze last 1-2 blocks + classifier:
Use when: moderate similarity, moderate dataset size (what we did)
Result: 93.07% on CIFAR-10

Strategy 3 — Unfreeze everything (full fine-tuning):
Use when: very different domain, large dataset available
Risk: catastrophic forgetting if learning rate too high, expensive
```

We unfroze `layer4` (last residual block) because:
```
Layers 1-3 → universal features (edges, textures, shapes) → keep frozen
Layer 4    → ImageNet specific high level features → needs adapting for CIFAR-10
FC         → 1000 ImageNet classes → replace entirely with 10 CIFAR-10 classes
```

#### The Preprocessing Contract — What Pretrained Models Expect

Using a pretrained model incorrectly is worse than building from scratch:

```
Wrong config (our first attempt):  40.26%  ← worse than random for some classes
Right config (fixed):              93.07%  ← 11% better than custom CNN

The difference: two preprocessing lines
```

Every pretrained model has a **preprocessing contract** — specific input size and normalization it was trained with:

```
ResNet18 contract:
Input size:    224×224 pixels
Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

Violate this contract → model receives inputs it has never seen → features meaningless
```

Always check the original paper or model card for preprocessing requirements before fine-tuning. This is non-negotiable.

### 6. The Mistake That Cost 53% Accuracy

First ResNet attempt scored 40% — worse than our custom CNN.

**Two mistakes:**

```
Mistake 1 — Wrong image size:
ResNet designed for: 224×224 pixels
CIFAR-10 images:     32×32 pixels
Features get destroyed by pooling before deep layers can learn

Mistake 2 — Wrong normalization:
ResNet expects: mean=[0.485, 0.456, 0.406] std=[0.229, 0.224, 0.225]
We used:        mean=[0.4914, 0.4822, 0.4465] std=[0.2023, 0.1994, 0.2010]
Mismatch between what model expects and what it receives
```

**Fix:**
```python
transforms.Resize(224)              # resize to correct input size
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],     # ImageNet normalization
    std=[0.229, 0.224, 0.225]
)
```

Result: 40% → 93% from fixing two preprocessing lines.

**This is one of the most common production mistakes with pretrained models.**

### 7. Fine-tuning Strategy — Unfreeze Last Block

Only freezing all pretrained layers and training the final classifier gave poor results on small CIFAR images. Better approach:

```python
# Unfreeze layer4 (last residual block) + final classifier
for name, param in resnet_fixed.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

Layer4 was trained for ImageNet's specific classes. CIFAR-10 has different classes — layer4 needs to adapt. Layers 1-3 capture universal features (edges, textures) that work for any images — keep them frozen.

### 8. Learning Rate For Fine-tuning

```python
lr=0.0001   # 10x lower than training from scratch (0.001)
```

Pretrained weights are already good — large updates destroy them (catastrophic forgetting). Low learning rate preserves pretrained knowledge while gently adapting to new task.

---

## Data Augmentation vs Transfer Learning

| Approach | Accuracy | Data Needed | Training Time | Best For |
|---|---|---|---|---|
| CNN from scratch | 82% | 50,000+ images | Long | Full control, custom architecture |
| Augmentation | 78%* | 50,000 images | Longer | Improving existing models |
| Transfer learning | 93% | Even 1,000 images | Short | Most production use cases |

*Needs 40+ epochs to show true benefit

---

## Tools & Libraries

```python
datasets              # Hugging Face dataset loading
numpy                 # Numerical operations
torch                 # PyTorch deep learning
torch.nn              # Neural network layers
torchvision.models    # Pretrained models (ResNet18)
torchvision.transforms # Image augmentation pipeline
sklearn               # Evaluation metrics
matplotlib            # Visualisation
python-dotenv         # Environment variable management
huggingface_hub       # Authentication
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/shambhavichaugule/cnn.git
cd cnn

# Activate virtual environment
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Add your Hugging Face token to .env
echo "HF_TOKEN=your_token_here" > .env

# Train CNN from scratch
python cnn.py

# Run evaluation only
python evaluate.py

# Train with augmentation and transfer learning
python data_augmentation.py
```

---

## PM Perspective

### Data Augmentation Is A Cost Decision Before It Is A Technical One

Before asking your data scientist to implement augmentation, ask these questions:

**Is our model overfitting?**
```
Training accuracy >> Test accuracy → yes, augmentation will help
Training accuracy ≈ Test accuracy → model is generalising, augmentation may not help
```

**What variations will real users introduce?**
```
User uploads product photos from phones → vary in lighting, angle, background → augment
User scans standardised forms on a fixed scanner → consistent inputs → augmentation may not help
```

**Do we have enough data?**
```
< 5,000 images per class → augmentation is essential, not optional
> 100,000 images per class → augmentation gives marginal benefit
```

**Can we afford the training time?**
```
Augmentation needs 2x more epochs to converge
If GPU costs $5/hour and training takes 10 hours → augmentation adds $50
Worth it for production models, may not be for quick experiments
```

The PM owns this cost-benefit calculation. Data scientists will implement whatever they are asked to — but the decision of whether augmentation is worth the training time and compute cost is a product decision.

### Transfer Learning Is A Build vs Buy Decision

Every time your team faces an image classification problem the first question should not be "what architecture should we build?" It should be "what pretrained model can we fine-tune?"

```
Build from scratch:
- Needs 100,000+ images to match pretrained performance
- Weeks of training and iteration
- Requires senior ML engineer to design architecture
- High compute cost
- Full control over what model learns

Fine-tune pretrained:
- Works with 1,000 images
- Hours to production
- Junior ML engineer can execute
- Low compute cost
- Less control, but rarely needed
```

The only reasons to build from scratch are:
- Highly specialised domain where no pretrained model exists
- You have millions of domain specific images and time to train
- Regulatory requirements demand full model provenance

For 95% of product use cases — fine-tune a pretrained model.

### The Preprocessing Contract Is A PM Responsibility

```
Wrong preprocessing: 40%
Right preprocessing: 93%
Gap:                 53%
```

This 53% gap happened silently. No error was thrown. The model trained for 10 epochs, produced a number, and looked like it was working. In production this would have shipped as a broken product.

The PM must ensure:

**Before model selection:** Document the preprocessing requirements of any pretrained model before the team starts implementation. This is in the model card, the original paper, or the framework documentation. It takes 10 minutes to read and prevents days of debugging.

**Before evaluation:** Define a minimum accuracy baseline that must be met before any model is shipped. A 40% accuracy model on a 10-class problem is barely better than random — there should be a gate that catches this.

**Before deployment:** Validate that the preprocessing pipeline in production exactly matches the preprocessing pipeline during training. This is a common source of production failures where training accuracy is 93% but production accuracy is 60%.

### When Augmentation and Transfer Learning Work Best Together

```
Transfer learning alone: 93.07%
Augmentation alone (20 epochs): 78.13% ← needs more epochs
Transfer learning + augmentation (40+ epochs): likely 94-95%+
```

In production the right answer is almost always both:
- Transfer learning provides the knowledge base
- Augmentation teaches the model to handle real world input variations

The combination is especially powerful when:
- Your training data comes from a controlled environment (studio photos)
- But production data comes from the wild (user phone cameras)
- Augmentation bridges that distribution gap

### What To Monitor After Launch

**Input distribution drift** — a model trained on well-lit studio photos fails on dark, blurry phone camera shots. Monitor the distribution of input image brightness, contrast, and resolution.

**Class distribution shift** — if cat images suddenly become 40% of requests but were only 10% of training data, overall accuracy may stay constant while cat accuracy degrades. Monitor per-class accuracy separately.

**Confidence calibration** — monitor the distribution of softmax probabilities. If average confidence drops over time the model is becoming less certain — early warning signal before accuracy drops.

**Augmentation coverage** — periodically review whether your augmentation pipeline still reflects real world input variations. If users start submitting a new type of image your augmentations didn't cover, add it.

### Key Insight

> The question is never "should we use augmentation?" or "should we use transfer learning?" The question is "what does our specific data, timeline, and accuracy requirement call for?" Both techniques have clear use cases and clear anti-use cases. Knowing the difference is what separates a PM who ships good AI products from one who ships models that looked great in the demo.

Using a pretrained model incorrectly (40%) is worse than building from scratch (82%). Evaluating augmentation too early (78% at 20 epochs) makes it look like it failed when it actually needed more time. Both are PM failures as much as engineering failures — because the PM must define evaluation criteria, not just read the final number.

---
