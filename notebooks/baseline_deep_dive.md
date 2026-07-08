# Baseline Model — Deep Dive & Full Repo Context

## Repo Overview

| Phase | Status | What it covers |
|-------|--------|---------------|
| Phase 0 — Setup | Done | Repo structure, env, dependencies |
| Phase 1 — Data Pipeline | Done | EDA, dataset class, dataloaders, stratified split |
| Phase 2 — Baseline | Done | Baseline CNN trained, 5.04% val accuracy |
| Phase 3–5 | TODO | Transfer learning, analysis, report |

---

## Project Structure

```
src/
  models/baseline.py        # Baseline CNN (2-layer)
  dataloaders/dataset.py     # Dataset class, transforms, dataloaders
  train.py                   # Training script (loop, wandb logging, checkpointing)
  utils/                     # (empty, for future helpers)
notebooks/
  01_eda.ipynb               # Exploratory Data Analysis
configs/                     # (empty, for future configs)
data/
  labels.csv                 # 10,222 rows: image_id → breed
  train/                     # 10,222 JPEG images
  test/                      # 10,357 JPEG images
checkpoints/                 # Saved model weights (.pth)
```

---

## Deep Dive: Baseline Model (`src/models/baseline.py`)

### Architecture — layer by layer

Input: a batch of images, shape `(B, 3, 224, 224)` — B images, 3 color channels (RGB), 224x224 pixels.

**Layer 1: `Conv2d(3, 16, kernel_size=3, stride=1, padding=1)` + ReLU + MaxPool2d(2)**
- Takes 3 input channels (RGB), outputs 16 feature maps
- `kernel_size=3`: each filter is a 3x3 window sliding across the image
- `stride=1`: moves 1 pixel at a time
- `padding=1`: adds 1 pixel of zeros around the border → output is same spatial size as input
- Output after conv: `(B, 16, 224, 224)`
- ReLU: `max(0, x)` — kills negative values, introduces non-linearity
- MaxPool2d(2,2): takes the max of each 2x2 window → halves spatial dimensions
- **Output: `(B, 16, 112, 112)`**
- **Parameters**: 16 filters × (3×3×3 weights + 1 bias) = 16 × 28 = **448 params**

**Layer 2: `Conv2d(16, 32, kernel_size=3, stride=1, padding=1)` + ReLU + AdaptiveAvgPool2d(1)**
- Takes 16 channels, outputs 32 feature maps
- Same 3x3 kernel with padding=1 → preserves spatial size
- Output after conv: `(B, 32, 112, 112)`
- ReLU again
- **AdaptiveAvgPool2d(1)**: this is the key move — it averages each entire feature map down to a single value. Regardless of input size, output is always `(B, 32, 1, 1)`
- This is why the model works with any input resolution (not just 224x224)
- **Output: `(B, 32, 1, 1)`**
- **Parameters**: 32 filters × (3×3×16 weights + 1 bias) = 32 × 145 = **4,640 params**

**Flatten: `x.view(x.size(0), -1)`**
- Reshapes from `(B, 32, 1, 1)` to `(B, 32)` — just removes the spatial dimensions

**FC1: `Linear(32, 128)` + ReLU**
- Fully connected layer: every one of the 32 features connects to 128 neurons
- **Parameters**: 32×128 + 128 bias = **4,224 params**

**FC2: `Linear(128, num_classes)` — the classifier head**
- 128 → 120 (one output per breed)
- **Parameters**: 128×120 + 120 bias = **15,480 params**
- Output: raw logits (unnormalized scores) — `CrossEntropyLoss` applies softmax internally

### Total Parameter Count

| Layer | Parameters |
|-------|-----------|
| conv1 (3→16) | 448 |
| conv2 (16→32) | 4,640 |
| fc1 (32→128) | 4,224 |
| fc2 (128→120) | 15,480 |
| **Total** | **~24,792** |

### Tensor Shape Flow

```
Input:          (B, 3, 224, 224)
After conv1:    (B, 16, 224, 224)
After maxpool1: (B, 16, 112, 112)
After conv2:    (B, 32, 112, 112)
After avgpool:  (B, 32, 1, 1)
After flatten:  (B, 32)
After fc1:      (B, 128)
After fc2:      (B, 120)        ← logits (one per breed)
```

---

## Key Concepts — Potential Exam / Professor Questions

**Q: Why `padding=1` with `kernel_size=3`?**
→ Preserves spatial dimensions. Without padding, each 3x3 conv shrinks the image by 2 pixels in each direction. `padding=1` adds a 1-pixel border of zeros so output = input size.

**Q: What does `AdaptiveAvgPool2d(1)` do and why use it instead of a second MaxPool?**
→ It global-average-pools each feature map to a single scalar. It makes the model input-size-agnostic (works with any resolution). A regular MaxPool would keep spatial structure, requiring you to calculate the exact flattened size for the FC layer.

**Q: Why is there no softmax at the end?**
→ `nn.CrossEntropyLoss` combines `LogSoftmax + NLLLoss` internally. Adding softmax would apply it twice, which is wrong. The model outputs raw logits.

**Q: Why does this model perform poorly (5% accuracy)?**
→ Only 2 conv layers and ~25K params. Dog breeds are fine-grained (malinois vs german shepherd look nearly identical). The model has too little capacity and receptive field to learn discriminative features. Pretrained models (ResNet, EfficientNet) have 100-1000x more parameters and were trained on millions of images.

**Q: What is the receptive field?**
→ After conv1 (3x3) + maxpool(2) + conv2 (3x3), each neuron in conv2 "sees" about a 10x10 pixel region of the original image. That's tiny — the model can only learn very local texture patterns, not global structure like ear shape or body proportions.

**Q: `model.train()` vs `model.eval()` — what changes?**
→ Affects layers like Dropout and BatchNorm (this model has neither, but it's good practice). `train()` enables stochastic behavior, `eval()` disables it. `@torch.no_grad()` on validation additionally skips gradient computation → saves memory & speed.

**Q: Why Adam optimizer?**
→ Adam = adaptive learning rate per parameter. Combines momentum (SGD+momentum) with RMSprop (scales by running average of gradient magnitudes). Works well out-of-the-box, especially with lr=1e-3.

**Q: What does `loss.item() * images.size(0)` do in the loss accumulation?**
→ `criterion` returns the mean loss over the batch by default. Multiplying by batch size recovers the total loss for that batch. Then dividing `running_loss / total` at the end gives a proper weighted average across batches (important because the last batch may be smaller).

**Q: What does the checkpoint save?**
→ `model_state_dict` (learned weights), `optimizer_state_dict` (Adam's momentum buffers — needed to resume training without resetting), epoch, val metrics, and the breed→index mapping.

---

## Training Configuration (`src/train.py`)

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Loss function | CrossEntropyLoss |
| Epochs | 20 |
| Batch size | 32 |
| Val split | 20% |
| Device | MPS (Apple Silicon) > CUDA > CPU |
| Logging | Weights & Biases (wandb) |

---

## Data Pipeline Summary (`src/dataloaders/dataset.py`)

- **10,222 training images**, split 80/20 → 8,177 train / 2,045 val
- **120 breeds**, 66–126 samples each (mean 85, mild imbalance)
- **Stratified split**: preserves breed proportions in both sets
- **Train augmentation**: RandomResizedCrop(224), RandomHorizontalFlip, ColorJitter
- **Val transform**: Resize(256) → CenterCrop(224) — deterministic, no randomness
- **Normalization**: ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`
- **DataLoader**: batch_size=32, shuffle=True (train), pin_memory=True (faster GPU transfer)

---

## Baseline Results

| Metric | Value |
|--------|-------|
| Best val accuracy | **5.04%** (epoch 17/20) |
| Best val loss | 4.4487 |
| Model params | ~25K |
| Random chance | 0.83% (1/120) |

The baseline learns something (5% vs 0.8% random), but is far too shallow and small to distinguish 120 visually similar breeds. This is the benchmark to beat in Phase 3.

---

## Why the Baseline Fails — and What Phase 3 Needs

1. **Too few parameters** (~25K vs millions in modern CNNs)
2. **Too shallow** (2 conv layers → tiny receptive field)
3. **No pretrained features** (learning from scratch with only 10K images)
4. **No regularization** (no dropout, no weight decay, no batch norm)
5. **No learning rate schedule** (constant lr throughout training)

Phase 3 solutions: transfer learning (ResNet50/EfficientNet), dropout, weight decay, label smoothing, cosine annealing LR, advanced augmentation (mixup, cutmix).
