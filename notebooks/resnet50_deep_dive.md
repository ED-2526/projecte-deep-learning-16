# Model 1: ResNet50 Frozen Backbone — Deep Dive

## What Are We Building?

Take a ResNet50 that was already trained on ImageNet (1.2M images, 1000 classes — including many dog breeds), freeze all its layers, and train only a small classifier head on our 120 dog breeds.

Think of it like this: ResNet50 already knows how to "see" — edges, textures, shapes, ears, fur patterns. We just teach it to map what it sees to our 120 breed labels.

---

## Architecture

```
Input image (B, 3, 224, 224)
        │
        ▼
┌──────────────────────────────┐
│     ResNet50 Backbone        │  ← FROZEN (23.5M params, requires_grad=False)
│                              │
│  conv1 (7x7, 64 filters)     │  Layer 0: initial convolution
│  bn1 + relu + maxpool        │
│  layer1 (3 bottleneck blocks)│  64→256 channels
│  layer2 (4 bottleneck blocks)│  256→512 channels
│  layer3 (6 bottleneck blocks)│  512→1024 channels
│  layer4 (3 bottleneck blocks)│  1024→2048 channels
│  AdaptiveAvgPool2d(1,1)      │  Global average pool → (B, 2048)
└──────────────────────────────┘
        │
        ▼ feature vector: (B, 2048)
        │
┌──────────────────────────────┐
│     Classifier Head          │  ← TRAINABLE (~1.1M params)
│                              │
│  Linear(2048, 512)           │  Bottleneck projection
│  ReLU()                      │  Non-linearity
│  Dropout(0.5)                │  Regularization
│  Linear(512, 120)            │  Class logits
└──────────────────────────────┘
        │
        ▼ output: (B, 120) raw logits
```

### Parameter Count

| Component | Parameters | Trainable? |
|-----------|-----------|------------|
| ResNet50 backbone | 23,508,032 | No (frozen) |
| Linear(2048, 512) + bias | 1,049,088 | Yes |
| Linear(512, 120) + bias | 61,560 | Yes |
| **Total** | **24,618,680** | **1,110,648 trainable** |

Only 4.5% of the model's parameters are actually trained. The rest are frozen ImageNet knowledge.

---

## Why ResNet50?

### Why not ResNet18?
- ResNet18 produces 512-d features. ResNet50 produces 2048-d features.
- For fine-grained classification (120 visually similar dog breeds), you need a richer feature space. The difference between a Malinois and a German Shepherd might be encoded in a few specific dimensions — with only 512 you may not have enough room.
- Empirically: ResNet50 outperforms ResNet18 by 5-8% on Stanford Dogs (a very similar benchmark to ours).

### Why not ResNet101 or ResNet152?
- Diminishing returns. ResNet101 gives only 1-2% more accuracy than ResNet50 while doubling compute.
- For a first transfer learning model, ResNet50 is the sweet spot on the effort-vs-accuracy curve.
- Worth testing later as an ablation, but not the starting point.

### Why not VGG16?
- 138M parameters (5x ResNet50), much slower, much more memory.
- No skip connections → worse gradient flow → features that transfer worse.
- Legacy architecture. No reason to use it in 2026.

### Why not EfficientNet?
- EfficientNet is actually excellent and should be Model 2.
- But ResNet50 is better as Model 1 because: (a) it's the most studied architecture for transfer learning, (b) every paper benchmarks against it, making our results directly comparable, (c) its architecture (residual blocks + skip connections) is simpler to explain to a professor than EfficientNet's compound scaling + squeeze-and-excitation blocks.

---

## Every Hyperparameter Decision — The "Why" Behind Each Choice

### Optimizer: Adam

**Chosen value:** `torch.optim.Adam(model.trainable_params(), lr=1e-3)`

**Why Adam and not SGD?**
- We are only training the classifier head (~1.1M params) on fixed features. This is a relatively simple optimization problem — essentially training a small MLP.
- Adam has per-parameter adaptive learning rates. It converges faster (fewer epochs) and is less sensitive to the initial learning rate.
- SGD with momentum is better when fine-tuning the full backbone (the uniform step size acts as an implicit regularizer). But for a small head, that advantage disappears.
- SGD would need 2-3x more epochs to reach the same accuracy here.

**Why not AdamW?**
- AdamW = Adam with decoupled weight decay. It's the "better" Adam for regularization.
- But we're already using Dropout(0.5) as our regularizer, and we set weight_decay=0.
- AdamW would give essentially identical results to Adam when weight_decay=0.
- It's a good choice for Model 2 when we unfreeze the backbone, but overkill here.

### Learning Rate: 1e-3

**Chosen value:** `lr=0.001`

**Why this value?**
- The classifier head is randomly initialized (Xavier/Kaiming default). These weights start far from optimal and need relatively large updates to learn.
- 1e-3 is Adam's recommended default (from the original paper, Kingma & Ba 2015). It works reliably for training from scratch.

**Why not 1e-2 (higher)?**
- Too aggressive for Adam. The adaptive learning rates can spike, causing loss oscillation or even divergence. With 120 output classes, the softmax gradients can already be large — a high LR amplifies this.

**Why not 1e-4 (lower)?**
- Too conservative for randomly initialized weights. The head would need many more epochs (30-50+) to converge. 1e-4 is appropriate when fine-tuning pretrained weights (small adjustments), not when training from scratch (large movements needed).

**Why not 3e-4 ("Karpathy constant")?**
- Would also work fine, just slightly slower convergence. Would need ~5 more epochs. 1e-3 is faster and empirically works well.

### Loss Function: CrossEntropyLoss

**Chosen value:** `nn.CrossEntropyLoss()`

**Why CrossEntropyLoss?**
- Standard loss for multi-class classification. Combines LogSoftmax + Negative Log Likelihood.
- Well-understood, well-behaved gradients.
- Same as baseline → clean comparison (we want to isolate the impact of transfer learning, not change multiple variables at once).

**Why not Focal Loss?**
- Focal Loss was designed for extreme class imbalance (e.g., object detection where 99.9% of proposals are background).
- Our dataset has mild imbalance (66-126 samples per breed, ratio 1.9x). This is not the scenario Focal Loss was designed for.
- Adds a hyperparameter (gamma) that needs tuning. Unnecessary complexity.

**Why not Label Smoothing (CrossEntropyLoss(label_smoothing=0.1))?**
- Label smoothing is actually a good technique — it softens the targets from hard [0,0,1,0,...] to [0.0008, 0.0008, 0.9917, 0.0008,...], which prevents overconfident predictions.
- But it's a regularizer, and we want to isolate the effect of transfer learning. If we add label smoothing now, we can't tell if the improvement came from ResNet50 or from label smoothing.
- Worth testing as an ablation later (e.g., in Model 2 or Phase 4).

### Epochs: 15

**Chosen value:** 15 epochs

**Why 15?**
- With a frozen backbone, the head converges quickly. The features don't change (backbone is frozen), so the head just needs to learn a mapping from fixed 2048-d vectors to 120 classes.
- Typical convergence pattern:
  - Epochs 1-3: rapid jump (from random ~0.8% to 50%+)
  - Epochs 4-8: continued improvement (55-70%)
  - Epochs 8-15: marginal gains, potential overfitting starts
- Our training loop saves the best checkpoint by val accuracy, so going a bit over the optimal point is safe.

**Why not 20 (same as baseline)?**
- Could work, but likely wastes 5 epochs of compute where the model is overfitting or plateaued.
- The baseline needed 20 because it trained from scratch — much harder optimization. Feature extraction converges faster.

**Why not 10?**
- Might miss the last 1-2% of accuracy. 15 gives a safety margin.

### Batch Size: 32

**Chosen value:** 32 images per batch

**Why 32?**
- Same as baseline → apples-to-apples comparison.
- With frozen backbone, only the head's gradients are stored, so memory usage is low despite ResNet50's size.

**Could we use 64?**
- Yes, the M5 with MPS can handle it (frozen backbone uses ~3-4 GB at batch 64).
- Larger batches = more stable gradients, slightly faster training.
- But keeping 32 maintains comparability with baseline results.

### Weight Decay: 0.0 (none)

**Chosen value:** no weight decay

**Why no weight decay?**
- We already have Dropout(0.5) as regularization. Adding weight decay on top is double-regularizing.
- Weight decay (L2 regularization) penalizes large weights, encouraging the model to use all features equally.
- But in fine-grained classification, we WANT the model to focus heavily on the most discriminative features (e.g., ear shape for certain breeds). Penalizing large weights works against this.
- Weight decay matters more when fine-tuning millions of backbone params (prevents pretrained weights from drifting too far). For a small head, dropout is sufficient.

### LR Scheduler: None

**Chosen value:** constant learning rate throughout training

**Why no scheduler?**
- The head converges in 10-15 epochs on a well-behaved loss landscape (it's a small MLP on fixed features).
- Schedulers (CosineAnnealing, ReduceLROnPlateau) are most useful when:
  - Training for many epochs (50+) — not our case
  - Fine-tuning a large network where optimal LR changes as the model adapts — not our case
  - Hitting plateaus that need LR reduction to escape — unlikely for a small head

**When will we add a scheduler?**
- Model 2, when we unfreeze the backbone. Then CosineAnnealingLR is the standard choice.

### Dropout Rate: 0.5

**Chosen value:** `nn.Dropout(0.5)`

**Why 0.5?**
- During training, randomly zeroes 50% of the 512-d features in the head.
- Prevents the head from relying too heavily on any single feature → encourages redundant, robust representations.
- 0.5 is the theoretical optimum from Srivastava et al. (2014) — maximum entropy of the dropout mask → maximum regularization effect.

**Why not 0.3 (weaker)?**
- With only ~68 training images per class, overfitting is a real risk. 0.3 may not be enough regularization.

**Why not 0.7 (stronger)?**
- Dropping 70% of features makes it too hard for the head to learn. The model under-fits — not enough information flows through to learn the mapping.

### Head Bottleneck: 512 dimensions

**Chosen value:** `Linear(2048, 512)` — 4x compression

**Why a bottleneck at all?**
- ResNet50's 2048-d features are general-purpose (trained on ImageNet's 1000 classes including cars, furniture, landscapes). Many dimensions are irrelevant for dog breeds.
- The bottleneck forces the network to select and compress the most relevant features for our task.
- Without the bottleneck (just `Linear(2048, 120)`), you get a purely linear classifier. It can't learn non-linear decision boundaries, which matter for fine-grained tasks where breed differences are subtle.

**Why 512 and not 256?**
- 256 is an 8x compression — too aggressive. You risk losing discriminative information.

**Why 512 and not 1024?**
- 1024 is only a 2x compression — not enough to force feature selection. Also doubles the head's parameter count without proven benefit.

**Why 512 specifically?**
- 4x compression is a well-tested ratio in the literature. Balanced between information preservation and regularization.

---

## What "Freezing" Means in PyTorch

### The mechanics

```python
for param in self.backbone.parameters():
    param.requires_grad = False
```

This tells PyTorch: "do not compute gradients for these parameters." The effects:

1. **Forward pass: unchanged.** Images still pass through all ResNet50 layers. Every convolution, BatchNorm, and ReLU fires normally. Freezing does NOT skip computation — the backbone still processes images.

2. **Backward pass: gradients stop at the head-backbone boundary.** During `loss.backward()`, gradients flow backward through the head but stop when they hit the frozen backbone. No gradients are computed for the 23.5M backbone params.

3. **Optimizer step: only head params update.** We pass only trainable params to the optimizer: `Adam(model.trainable_params(), lr=1e-3)`. The backbone weights never change.

### Why this saves memory and time

- **Memory:** PyTorch normally stores intermediate activations during the forward pass (needed to compute gradients during backward). For frozen layers, it doesn't need to store these → ~60% memory savings compared to full fine-tuning.
- **Speed:** Backpropagation through the backbone is skipped entirely → ~2-3x faster than full fine-tuning.

### Why freeze for the first model?

1. **Establish a feature quality baseline.** We need to know: "Are ImageNet features good enough for dog breeds out of the box?" If frozen ResNet50 gives 70%, the features are already great. If 30%, they need adaptation. This informs whether we unfreeze later and how aggressively.

2. **Prevent overfitting.** 8,177 training images vs 23.5M backbone params. If we fine-tune everything, the model can memorize the training set. By freezing, we reduce trainable params from 23.5M to 1.1M.

3. **Fast iteration.** ~15 min on your Mac vs ~1+ hour for full fine-tuning. Lets us validate the pipeline and debug quickly.

---

## Why ImageNet Normalization is Mandatory

The dataset already applies:
```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

**Why these exact numbers?**
- These are the mean and standard deviation of the ImageNet training set's pixel values per channel (R, G, B).
- ResNet50's weights were trained on data normalized with these statistics. Every filter, every BatchNorm parameter assumes inputs are centered around 0 with std ~1.

**What happens with wrong normalization?**
- The first conv layer expects inputs around 0, not around 0.45. If you feed raw [0,1] images, activations are in the wrong range, BatchNorm statistics mismatch, and features become garbage.
- You might still get ~10-20% accuracy (the model is robust enough to partially compensate), but you leave 50%+ accuracy on the table.

**Why not compute our own dataset's mean/std?**
- The backbone doesn't care about our dataset's distribution. It was trained on ImageNet's distribution. Using our stats would shift the input and degrade feature quality.
- When we later unfreeze the backbone, it can gradually adapt — but for frozen, correct normalization is essential.

---

## Training Configuration Summary

| Parameter | Value | Why this? | Why not alternatives? |
|-----------|-------|-----------|----------------------|
| Backbone | ResNet50 (IMAGENET1K_V2) | Best-studied, 2048-d features, dogs in ImageNet | ResNet18 too small, ResNet101 diminishing returns |
| Backbone state | Frozen | Prevent overfitting, fast training, establish baseline | Fine-tuning comes in Model 2 |
| Head | Linear(2048→512) + ReLU + Dropout(0.5) + Linear(512→120) | Bottleneck for feature selection, non-linearity for fine-grained | Single Linear too limited, deeper head overfits |
| Optimizer | Adam | Fast convergence for small head | SGD needs 2-3x more epochs, AdamW redundant without weight decay |
| Learning rate | 1e-3 | Adam default, good for random init | 1e-2 diverges, 1e-4 too slow |
| Weight decay | 0.0 | Dropout is enough, weight decay hurts fine-grained focus | Would matter more when fine-tuning backbone |
| Scheduler | None | Head converges fast, loss landscape is smooth | Needed later for fine-tuning, not now |
| Loss | CrossEntropyLoss | Standard, same as baseline for clean comparison | Focal Loss for imbalanced (not our case), label smoothing is an ablation |
| Epochs | 15 | Converges by epoch 10-12, margin of safety | 20 wastes compute, 10 might miss tail gains |
| Batch size | 32 | Same as baseline for comparison | 64 possible but changes a variable |
| Dropout | 0.5 | Maximum regularization (Srivastava 2014) | 0.3 too weak for 68 imgs/class, 0.7 too aggressive |
| Bottleneck | 512 | 4x compression, standard ratio | 256 loses info, 1024 not enough compression |

---

## What to Expect

| Metric | Baseline (2-layer CNN) | ResNet50 Frozen (expected) |
|--------|----------------------|---------------------------|
| Val accuracy | 5.04% | **60-80%** |
| Trainable params | ~25K | ~1.1M |
| Total params | ~25K | ~24.6M |
| Training time | ~15 min | ~10-15 min |

The jump from 5% to 60-80% comes entirely from using pretrained features. The model already "knows" how to see — we just teach it our labels.

If we get <60%, that tells us the head design or hyperparams need tweaking.
If we get >75%, that's a strong baseline and fine-tuning (Model 2) should push it to 85%+.
