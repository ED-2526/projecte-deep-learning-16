# Iteration Log — Dog Breed Identification

This file tracks every model iteration: what changed, why, what each parameter does, and what the results tell us. The goal is full traceability — if a professor asks "why did you set dropout to 0.3?", the answer is here.

---

## Data Pipeline (shared across all iterations)

Before any model runs, we built the data pipeline. Understanding it is important because **the same data flows into every model** — changes here affect all results.

### Dataset

In supervised learning, a **dataset** is a collection of (input, label) pairs — in our case, (dog image, breed name). The model learns by seeing many examples and adjusting its parameters to predict the correct label from the input. The dataset is split into two non-overlapping subsets:

- **Training set** — the model learns from these images. It sees them repeatedly (once per epoch) and updates its weights to reduce prediction errors on them.
- **Validation set** — images the model never trains on. After each epoch, we evaluate on this set to measure how well the model generalizes to unseen data. If train accuracy is high but val accuracy is low, the model is **overfitting** (memorizing the training set instead of learning general patterns).

A **stratified split** ensures that each breed appears in the same proportion in both sets. Without stratification, random chance might put most images of a rare breed in the training set and none in validation, making the val score unreliable for that breed.

- **Source**: Kaggle Dog Breed Identification — 10,222 labeled training images, 120 breeds
- **Split**: stratified 80/20 → 8,177 train / 2,045 val *(dataset.py:120-125)* (stratified = each breed has the same proportion in train and val, so no breed is over/under-represented in either set)
- **Breed distribution**: 66–126 images per breed (mean ~85). Mild imbalance (ratio 1.9x between rarest and most common), not extreme enough to need special handling like oversampling or focal loss.

### Transforms

A **transform** is any operation applied to an image before it enters the model. Transforms serve two purposes:

1. **Preprocessing** — converting raw images into the format the model expects. Images come from disk as JPEGs of varying sizes and pixel ranges (0–255). Models need fixed-size tensors with normalized values. Transforms like `Resize`, `ToTensor`, and `Normalize` handle this conversion. These are applied to both training and validation data identically.

2. **Data augmentation** — artificially creating variations of each training image (crops, flips, color shifts) so the model sees a different version every epoch. This effectively multiplies the dataset size and forces the model to learn features that are invariant to these changes (e.g., "this is a golden retriever regardless of lighting or whether I see the full body or just the head"). Augmentation is applied **only during training** — validation must be deterministic so results are comparable across epochs.

Transforms are chained into a pipeline using `transforms.Compose([...])` *(dataset.py:66-72)* — each image passes through every transform in order, left to right.

**Training transforms** (applied every time an image is loaded during training — each epoch sees slightly different versions of the same image):

| Transform | Code | What it does | Why |
|-----------|------|-------------|-----|
| `RandomResizedCrop(224, scale=(0.8, 1.0))` | *dataset.py:67* | Randomly crops 80-100% of the image area, then resizes to 224x224 | Forces the model to recognize breeds from partial views and different scales. A dog's ear alone should still help classify it. |
| | | `224` — target output size in pixels (square). Every crop is resized to exactly 224x224, regardless of original image dimensions. | |
| | | `scale=(0.8, 1.0)` — crop area is randomly chosen between 80% and 100% of the original. 0.8 means sometimes only 80% of the image is kept (slight zoom-in). Lower values = more aggressive crops. | |
| `RandomHorizontalFlip()` | *dataset.py:68* | 50% chance of mirroring the image left-right | Dogs look the same mirrored. Doubles effective dataset size for free. Vertical flip would be wrong — dogs don't appear upside down. |
| | | Default `p=0.5` — each image has a 50/50 chance of being flipped. This default is almost always used. | |
| `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)` | *dataset.py:69* | Randomly adjusts brightness, contrast, and saturation by up to ±20% | Makes the model robust to lighting conditions. A golden retriever in shade vs sunlight is still a golden retriever. |
| | | `brightness=0.2` — randomly multiplies brightness by a factor in [0.8, 1.2]. 0 = no change, 1 = extreme variation. | |
| | | `contrast=0.2` — same range for contrast (difference between light and dark areas). | |
| | | `saturation=0.2` — same range for color intensity. 0 saturation = grayscale. | |
| `ToTensor()` | *dataset.py:70* | Converts PIL image (H,W,C) with values 0-255 to PyTorch tensor (C,H,W) with values 0.0-1.0 | PyTorch models expect float tensors in channel-first format. |
| | | No parameters. Divides pixel values by 255 (0-255 → 0.0-1.0) and reorders dimensions from (Height, Width, Channels) to (Channels, Height, Width). | |
| `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` | *dataset.py:71* | Subtracts ImageNet mean and divides by ImageNet std, per channel | Centers pixel values around 0. These specific numbers are ImageNet statistics — mandatory when using pretrained models (their weights expect this distribution). |
| | | `mean=[0.485, 0.456, 0.406]` — per-channel mean to subtract (Red, Green, Blue). These are the average pixel values across all 1.2M ImageNet images. | |
| | | `std=[0.229, 0.224, 0.225]` — per-channel standard deviation to divide by. After normalization: each channel has mean near 0 and std near 1. | |

**Validation transforms** (deterministic, no randomness — so evaluation is reproducible):

| Transform | Code | What it does | Why |
|-----------|------|-------------|-----|
| `Resize(256)` | *dataset.py:75* | Resizes shortest side to 256px (preserving aspect ratio) | Slightly larger than 224 so we can center-crop without losing content at edges. |
| | | `256` — target size for the shortest edge. A 400x600 image becomes 256x384. Computed as `int(224 * 1.14)`. | |
| `CenterCrop(224)` | *dataset.py:76* | Takes the center 224x224 region | Deterministic crop — always the same region. No randomness so val results are comparable across epochs. |
| | | `224` — output size. Cuts equally from all four sides, keeping the center. A 256x384 image becomes 224x224 (16px cut from top/bottom, 80px from left/right). | |
| `ToTensor()` + `Normalize(...)` | *dataset.py:77-78* | Same as training (see parameter details above) | Same preprocessing so the model sees consistently formatted inputs. |

**Why these augmentations and not others?**
- No `RandomRotation`: dogs can appear at slight angles, but extreme rotation (45°+) creates unrealistic images with black corners. Not worth the noise.
- No `RandomErasing`/`Cutout`: these are stronger augmentations useful when overfitting is severe. Better to add later if needed, not from the start.
- No `Mixup`/`CutMix`: these blend two images and their labels. Powerful but changes the loss computation. Better to add as a separate iteration so we can measure their impact cleanly.

### DataLoader

A **DataLoader** wraps a Dataset and handles the logistics of feeding data to the model during training. Without it, you'd have to manually: load images one by one, group them into batches, shuffle the order each epoch, and manage parallel loading. The DataLoader does all of this automatically.

The key idea is **batching**: instead of showing the model one image at a time (slow, noisy gradients), we show it a batch of images (e.g., 32) simultaneously. The model processes all 32, computes the average loss across them, and makes one gradient update. This is faster (GPU parallelism) and more stable (averaging over 32 samples reduces noise in the gradient estimate).

| Parameter | Code | Value | Why |
|-----------|------|-------|-----|
| `batch_size` | *dataset.py:137, train.py:34* | 32 | Standard starting point. Small enough to fit in memory on any GPU/MPS device. Large enough for stable gradient estimates. |
| | | | Number of images grouped together per training step. Larger batches = more stable gradients but more memory. Common values: 16, 32, 64. |
| `shuffle` | *dataset.py:138,145* | True (train) / False (val) | Training: random order each epoch prevents the model from learning sequence patterns. Validation: fixed order for reproducibility. |
| | | | When True, the DataLoader randomly reorders all samples at the start of each epoch. This prevents the model from learning patterns in the data order (e.g., all terriers appearing first). |
| `num_workers` | *dataset.py:139* | 4 | Parallel data loading — 4 CPU threads prepare the next batches while the GPU processes the current one. Prevents the GPU from waiting for data. |
| | | | Number of separate CPU processes that load and transform images in the background. 0 = load in the main process (slow). 4 = four parallel workers pre-fetch upcoming batches. |
| `pin_memory` | *dataset.py:140* | True | Pre-allocates tensors in page-locked (pinned) RAM, which enables faster CPU→GPU transfer. Standard practice. |
| | | | When True, tensors are stored in non-swappable RAM, allowing faster DMA (Direct Memory Access) transfer to GPU. Always set True when using GPU/MPS. |

---

## Iteration 0: Baseline CNN

**Purpose**: establish a lower bound. This is the professor's starting point — a deliberately simple model to show that even a minimal CNN learns *something*, but is nowhere near sufficient for fine-grained classification.

### Architecture *(src/models/baseline.py)*

```
Input: (B, 3, 224, 224)

Conv2d(3→16, kernel=3, stride=1, padding=1)     # 16 filters detect basic patterns (edges, color blobs)
ReLU()                                            # kills negative activations, introduces non-linearity
MaxPool2d(kernel=2, stride=2)                     # halves spatial dims → (B, 16, 112, 112)

Conv2d(16→32, kernel=3, stride=1, padding=1)     # 32 filters detect slightly more complex patterns
ReLU()
AdaptiveAvgPool2d(1)                              # global average pool → (B, 32, 1, 1)

Flatten → (B, 32)
Linear(32→128) + ReLU                            # small fully-connected layer
Linear(128→120)                                  # one output per breed (raw logits)
```

### Layer-by-layer explanation

A CNN (Convolutional Neural Network) is built from a few types of building blocks stacked in sequence. Each block transforms the data in a specific way. Here's what each type does conceptually:

- **Convolutional layers** (`Conv2d`) — the core of a CNN. A convolutional layer slides small learnable filters (e.g., 3x3 pixels) across the image. Each filter detects a specific pattern: early layers learn edges and color gradients, deeper layers learn complex shapes like ears or eyes. The output is a set of **feature maps** — one per filter — where bright spots indicate "this pattern was found here."

- **Activation functions** (`ReLU`) — applied after each layer to introduce non-linearity. Without them, stacking many layers would be mathematically equivalent to a single layer (linear operations compose into linear operations). ReLU (`max(0, x)`) is the simplest and most common: it keeps positive values and zeroes out negatives.

- **Pooling layers** (`MaxPool2d`, `AdaptiveAvgPool2d`) — reduce the spatial dimensions of feature maps. This serves two purposes: (1) reduces computation (fewer pixels to process in subsequent layers), and (2) introduces some translation invariance (the exact position of a feature matters less — a dog's ear detected at pixel 50 or 52 should produce the same output). MaxPool keeps the strongest activation in each region; AvgPool takes the mean.

- **Fully connected layers** (`Linear`) — standard neural network layers where every input connects to every output. Used at the end of the network to combine all the spatial features into a final classification decision. The last FC layer has one output per class (120 for our breeds).

- **Flatten** (`x.view(...)`) — reshapes a multi-dimensional tensor (e.g., 32 feature maps of 1x1) into a flat vector (e.g., 32 values). Bridges the gap between convolutional layers (which work on spatial grids) and FC layers (which expect flat vectors).

**Conv2d(3, 16, kernel_size=3, stride=1, padding=1)** *(baseline.py:9)*
- 16 learnable 3x3 filters scan across the image. Each filter produces one feature map.
- `3` — input channels. RGB images have 3 channels (Red, Green, Blue).
- `16` — output channels (number of filters). Each filter learns to detect a different pattern. More filters = more patterns detected, but more parameters.
- `kernel_size=3` — each filter is a 3x3 pixel window. Small enough to detect local patterns (edges, corners), large enough to capture meaningful structure. 3x3 is the most common choice in modern CNNs.
- `stride=1` — the filter moves 1 pixel at a time. Every position is evaluated — maximum resolution, no skipping.
- `padding=1` — adds a 1-pixel border of zeros around the image before applying the filter. With kernel_size=3 and padding=1, the output has the same spatial size as the input. Without padding, each conv layer would shrink the image by 2 pixels per dimension.
- Output: (B, 16, 224, 224) — 16 feature maps, same spatial size.
- **Parameters**: 16 filters x (3x3x3 weights + 1 bias) = 16 x 28 = 448

**ReLU (Rectified Linear Unit)** *(baseline.py:10)*
- `f(x) = max(0, x)` — zeroes out negative values, keeps positive ones unchanged.
- No learnable parameters. It's a fixed mathematical function applied element-wise to every value in the tensor.
- Without non-linearity, stacking linear layers (conv is linear) would collapse into a single linear operation. ReLU lets the network learn non-linear relationships.

**MaxPool2d(kernel_size=2, stride=2)** *(baseline.py:11)*
- Takes the maximum value in each 2x2 window, halving spatial dimensions (224 to 112).
- `kernel_size=2` — the pooling window is 2x2 pixels. Each window produces one output value (the maximum).
- `stride=2` — the window moves 2 pixels at a time (no overlap). Combined with kernel_size=2, this exactly halves width and height.
- Keeps the strongest activation in each region — a form of translation invariance (the exact position of a feature matters less).
- No learnable parameters — it's a fixed operation.
- Output: (B, 16, 112, 112)

**Conv2d(16, 32, kernel_size=3, stride=1, padding=1)** *(baseline.py:13)*
- Same structure as conv1, but deeper.
- `16` — input channels: takes the 16 feature maps from layer 1.
- `32` — output channels: produces 32 higher-level feature maps. Each of the 32 filters looks at all 16 input channels — it learns combinations of the simpler patterns from layer 1.
- `kernel_size=3, stride=1, padding=1` — same settings as conv1: 3x3 filters, every position, size-preserving padding.
- Output: (B, 32, 112, 112)
- **Parameters**: 32 filters x (3x3x16 weights + 1 bias) = 32 x 145 = 4,640

**AdaptiveAvgPool2d(1)** *(baseline.py:15)*
- Averages each entire 112x112 feature map down to a single number.
- `1` — the target output size (1x1). Regardless of input spatial dimensions, each feature map is reduced to a single value by averaging all its pixels.
- Output: (B, 32, 1, 1) which is then flattened to (B, 32).
- This makes the model input-size agnostic — any resolution works, not just 224x224.
- Unlike a regular MaxPool (which would keep spatial structure), this collapses everything to a single vector, which is what the fully-connected layer expects.
- No learnable parameters.

**Flatten** *(baseline.py:30)*
- `x.view(x.size(0), -1)` reshapes from (B, 32, 1, 1) to (B, 32) — removes the spatial dimensions.
- `x.size(0)` — keeps the batch dimension unchanged. `-1` — PyTorch infers the remaining size automatically.

**Linear(32, 128) + ReLU** *(baseline.py:17-18)*
- Fully connected layer: every one of the 32 inputs connects to every one of the 128 outputs through a learnable weight.
- `32` — input features (matches the output of AdaptiveAvgPool2d).
- `128` — output features (number of neurons). A design choice — larger = more capacity but more parameters.
- Learns non-linear combinations of the averaged feature maps (non-linear because ReLU follows).
- **Parameters**: 32x128 weights + 128 biases = 4,224

**Linear(128, 120)** *(baseline.py:19)*
- The classifier head: maps 128 features to 120 class scores (one per breed).
- `128` — input features (matches previous layer's output).
- `120` — output features = `num_classes`. One output per breed. The highest value indicates the model's prediction.
- Output: raw logits (unnormalized scores). No softmax here — `CrossEntropyLoss` applies it internally.
- **Parameters**: 128x120 weights + 120 biases = 15,480

### Parameter count

| Layer | Parameters |
|-------|-----------|
| conv1 (3→16) | 448 |
| conv2 (16→32) | 4,640 |
| fc1 (32→128) | 4,224 |
| fc2 (128→120) | 15,480 |
| **Total** | **~24,792** |

### Training configuration

Training a neural network is an iterative optimization process. Each training step:

1. **Forward pass**: a batch of images flows through the model, producing predictions (logits).
2. **Loss computation**: a **loss function** compares those predictions against the true labels and outputs a single number measuring "how wrong" the model is. Lower = better.
3. **Backward pass** (backpropagation): PyTorch computes the gradient of the loss with respect to every learnable parameter — "if I nudge this weight up slightly, does the loss go up or down, and by how much?"
4. **Optimizer step**: the **optimizer** uses those gradients to update every parameter in the direction that reduces the loss. The **learning rate** controls how big each step is — too large and you overshoot the optimal point, too small and training takes forever.

One full pass through the entire training set is called an **epoch**. We train for multiple epochs, and each epoch the model sees every image once (in a different random order thanks to shuffling).

A **learning rate scheduler** optionally adjusts the learning rate during training — typically starting large (for fast initial progress) and decaying over time (for precise fine-tuning near the optimal point).

**Regularization** refers to any technique that prevents the model from overfitting (memorizing the training data instead of learning general patterns). Common forms include dropout (randomly disabling neurons), weight decay (penalizing large weights), and data augmentation (already covered above).

| Parameter | Code | Value | Why this value |
|-----------|------|-------|---------------|
| **Optimizer** | *train.py:148* | Adam | Adaptive learning rate per parameter. Converges faster than SGD for small models. Standard default. |
| | | | `Adam(params, lr)` — takes the model parameters and a learning rate. Internally maintains per-parameter momentum and scaling, so each weight gets updates proportional to its gradient history. |
| **Learning rate** | *train.py:36* | 1e-3 | Adam's recommended default (Kingma & Ba, 2015). Good starting point for training from scratch. |
| | | | Controls the step size for weight updates. `1e-3` = 0.001 — each weight changes by at most ~0.1% of the gradient per step. Higher = faster but riskier. Lower = safer but slower. |
| **Loss** | *train.py:147* | CrossEntropyLoss | Standard for multi-class classification. Combines LogSoftmax + NLL internally. |
| | | | Takes model outputs (120 raw scores) and the true label (a single integer 0-119). Applies softmax to convert scores into probabilities, then measures how wrong the predicted probability for the true class is. Perfect prediction = loss 0, random guessing on 120 classes = loss ~4.79. |
| **Epochs** | *train.py:35* | 20 | Enough for a small model to converge on this dataset. |
| | | | One epoch = one full pass through all 8,177 training images. 20 epochs means the model sees each image 20 times (with different augmentations each time). |
| **Batch size** | *train.py:34* | 32 | Standard. Fits in memory, gives reasonably stable gradients. |
| **Scheduler** | — | None | Simple model, short training — constant lr is fine. |
| **Regularization** | — | None | Intentionally absent. The baseline is meant to be minimal. |
| **Weight decay** | — | 0 | Same reason — keep it simple to establish a clean lower bound. |
| | | | Weight decay adds a penalty proportional to the magnitude of each weight to the loss. This discourages large weights. 0 = no penalty. |

### Results

| Metric | Value |
|--------|-------|
| Best val accuracy | **5.04%** (epoch 17/20) |
| Best val loss | 4.4487 |
| Random chance | 0.83% (1/120) |

### What the results tell us

- **5% vs 0.8% random**: the model learned *something*. It's 6x better than random guessing. But 5% is terrible for a real classifier.
- **Why so bad?**
  1. **Too few parameters** (~25K). Modern image classifiers have millions. 25K simply can't encode the visual complexity of 120 dog breeds.
  2. **Too shallow** (2 conv layers). Each neuron in the second conv layer "sees" roughly a 10×10 pixel region of the original 224×224 image. That's tiny — it can detect local textures (fur patterns, color patches) but not global structure (ear shape, body proportions, face geometry).
  3. **No pretrained knowledge**. The model starts from random weights and has only 8K training images to learn from. Modern pretrained models (ResNet, EfficientNet) were trained on 1.2M ImageNet images — they bring vast visual knowledge that this model lacks.
  4. **No regularization**. With no dropout or weight decay, the model likely overfits to the training set's small patterns rather than learning generalizable features.

- **The benchmark is set**: any improved model must beat 5.04%. This number also gives us a sanity check — if a future model gets *worse* than 5%, something is wrong with the pipeline, not the architecture.

---

## Iteration 1: ResNet50 Frozen Backbone

**Purpose**: test whether pretrained ImageNet features are sufficient for dog breed classification without modifying them at all. This isolates the value of transfer learning — the only thing we're training is a small classifier head on top of fixed features.

### What is transfer learning?

ResNet50 was trained on ImageNet (1.2M images, 1000 classes — including ~120 dog breeds among them). During that training, it learned to extract visual features: edges, textures, shapes, object parts, spatial relationships. These features are general-purpose — they're useful for any image task, not just ImageNet's 1000 classes.

Transfer learning reuses these features. Instead of learning to see from scratch (like the baseline), we take a model that already knows how to see and just teach it our specific labels.

**Frozen backbone** means we don't modify ResNet50's learned features at all. We only train a new classifier head that maps ResNet50's features to our 120 breeds. This is the simplest and safest form of transfer learning.

### Architecture *(src/models/transfer.py)*

```
Input: (B, 3, 224, 224)

┌─────────────────────────────────────────────────┐
│  ResNet50 Backbone — FROZEN (23.5M params)      │
│                                                  │
│  conv1: 7×7 conv, 64 filters, stride 2          │
│  bn1 + relu + maxpool(3,2)                       │
│  layer1: 3 bottleneck blocks (64→256 channels)   │
│  layer2: 4 bottleneck blocks (256→512 channels)  │
│  layer3: 6 bottleneck blocks (512→1024 channels) │
│  layer4: 3 bottleneck blocks (1024→2048 channels)│
│  AdaptiveAvgPool2d(1,1) → (B, 2048)             │
└─────────────────────────────────────────────────┘
            │
            ▼  feature vector: (B, 2048)
┌─────────────────────────────────────────────────┐
│  Classifier Head — TRAINABLE (~1.1M params)     │
│                                                  │
│  Linear(2048, 512)   bottleneck projection       │
│  ReLU()              non-linearity               │
│  Dropout(0.3)        regularization              │
│  Linear(512, 120)    class logits                │
└─────────────────────────────────────────────────┘
            │
            ▼  output: (B, 120) raw logits
```

### Why ResNet50 and not other architectures?

| Alternative | Why not (for this iteration) |
|-------------|-----|
| **ResNet18** | Produces 512-d features (vs 2048-d). For fine-grained classification where breed differences are subtle, you need a richer feature space. ResNet50 outperforms ResNet18 by 5-8% on Stanford Dogs (similar benchmark). |
| **ResNet101/152** | Diminishing returns. Only 1-2% more accuracy than ResNet50 while doubling compute. Not worth it as a starting point. |
| **VGG16** | 138M params (5x ResNet50), much slower, no skip connections → worse gradient flow and worse-transferring features. Legacy architecture. |
| **EfficientNet** | Actually excellent — should be a future iteration. But ResNet50 first because: (a) most studied for transfer learning, (b) every paper benchmarks against it making our results directly comparable, (c) simpler architecture to explain (residual blocks vs compound scaling + squeeze-excitation). |

### What "freezing" means in PyTorch

```python
# transfer.py:28-29
for param in self.backbone.parameters():
    param.requires_grad = False
```

This does three things:
1. **Forward pass: unchanged.** Images still pass through all 50 layers. Every convolution, BatchNorm, and ReLU fires normally. Freezing does NOT skip computation.
2. **Backward pass: gradients stop at the head.** During `loss.backward()`, gradients flow back through the head but stop when they hit the frozen backbone. No gradients computed for the 23.5M backbone params.
3. **Optimizer: only updates the head.** We pass `model.trainable_params()` *(transfer.py:43-45)* (only head params) to Adam *(train.py:142,148)*. Backbone weights never change.

**Why freeze?**
- **Prevent overfitting**: 8,177 training images vs 23.5M backbone params — if we fine-tune everything, the model can memorize the training set.
- **Establish a baseline for feature quality**: we need to know "how good are ImageNet features for dogs out of the box?" before deciding whether to fine-tune.
- **Speed**: ~15 min vs 1+ hour for full fine-tuning. Fast iteration, fast debugging.

### Classifier head design — every decision explained

**Linear(2048, 512) — the bottleneck** *(transfer.py:34)*
- ResNet50 outputs a 2048-dimensional feature vector. Many of those dimensions encode things irrelevant to dogs (cars, furniture, landscapes — ImageNet has 1000 diverse classes).
- The bottleneck compresses 2048→512 (4x reduction), forcing the network to select and combine the most relevant features for our task.
- Without this layer (just `Linear(2048, 120)`), you get a purely linear classifier. It can't learn non-linear decision boundaries, which matter when breed differences are subtle.
- **Why 512 and not 256?** 256 is 8x compression — too aggressive, risks losing discriminative info. **Why not 1024?** Only 2x compression — not enough to force feature selection, and doubles the head's parameter count without proven benefit. 512 (4x) is a well-tested ratio.

**ReLU()** *(transfer.py:35)*
- Non-linearity between the two linear layers. Without it, two stacked linear layers collapse into one (matrix multiplication is associative).
- Lets the head learn non-linear breed boundaries.

**Dropout(0.3)** *(transfer.py:36)*
- During training, randomly zeroes 30% of the 512 features in each forward pass.
- Prevents the head from relying too heavily on any single feature — forces it to build redundant, robust representations.
- **Why 0.3 and not 0.5?** With a frozen backbone, the head is already constrained (fixed features, only 1.1M trainable params — that's implicit regularization). 0.5 drops half the signal — too aggressive for fine-grained classification where subtle differences matter. 0.3 still regularizes but lets more information flow through.
- **Why not 0.1?** Too weak. With only ~68 training images per breed, overfitting is a real risk. Some regularization is needed.

**Linear(512, 120)** *(transfer.py:37)*
- Final classifier: maps the 512 compressed features to 120 breed scores (raw logits).
- CrossEntropyLoss applies softmax internally — no activation function here.

### Parameter count

| Component | Parameters | Trainable? |
|-----------|-----------|------------|
| ResNet50 backbone *(transfer.py:25)* | 23,508,032 | No (frozen) |
| Linear(2048→512) + bias *(transfer.py:34)* | 1,049,088 | Yes |
| Linear(512→120) + bias *(transfer.py:37)* | 61,560 | Yes |
| **Total** | **24,618,680** | **1,110,648 trainable** |

Only 4.5% of the model's parameters are actually trained. The other 95.5% are frozen ImageNet knowledge.

### Training configuration

| Parameter | Code | Value | Why this value | Why not alternatives |
|-----------|------|-------|---------------|---------------------|
| **Optimizer** | *train.py:148* | Adam | Fast convergence for a small head on fixed features. Per-parameter adaptive LR handles the different scales of head weights well. | SGD would need 2-3x more epochs. AdamW is redundant without weight decay. |
| **Learning rate** | *train.py:36* | 1e-3 | The head is randomly initialized (needs large updates). 1e-3 is Adam's default and works well for training from scratch. | 1e-2 too aggressive (Adam's adaptive LR can spike, causing oscillation). 1e-4 too conservative for random init (would need 30-50+ epochs). |
| **Loss** | *train.py:147* | CrossEntropyLoss | Standard multi-class loss. Same as baseline → clean comparison (isolate the impact of the architecture, not the loss). | Focal loss: designed for extreme imbalance (not our case). Label smoothing: good technique, but adding it here would confuse whether improvement came from ResNet50 or label smoothing. |
| **Epochs** | *train.py:35* | 50 | Enough to see the full convergence curve: where it peaks, when overfitting starts, when it plateaus. Best checkpoint is saved, so extra epochs cost only time. | 15 might leave accuracy on the table. 20 might still miss the tail. 50 gives full visibility. |
| **Batch size** | *train.py:34* | 32 | Same as baseline for clean comparison. Fits in memory with frozen backbone. | 64 is possible (frozen backbone uses less memory), but would change a variable vs baseline. |
| **Scheduler** | *train.py:149-151* | CosineAnnealingLR(T_max=50) | Smoothly decays lr from 1e-3 to ~0 over 50 epochs. Early epochs: large updates for fast learning. Late epochs: small updates for fine convergence. | No scheduler: flat lr wastes late epochs (too-large updates cause oscillation around the minimum). StepLR: sharp drops create jumps in the loss curve. ReduceLROnPlateau: reactive approach, but we know the landscape is smooth (small head on fixed features). |
| | | | `T_max=50` — the number of epochs over which the cosine curve completes one half-cycle (from max lr to ~0). Matches total training epochs so the lr reaches its minimum at the end of training. | |
| **Weight decay** | — | 0 | Already have Dropout for regularization. Weight decay penalizes large weights, which works against fine-grained classification (we WANT the model to focus strongly on the most discriminative features). | Would matter more when fine-tuning millions of backbone params. |
| **Dropout** | *transfer.py:36* | 0.3 | See head design section above. |  |
| | | | `0.3` — probability of each neuron being set to zero during training. 30% of the 512 features are randomly masked on each forward pass, forcing the network to not depend on any single feature. At evaluation time, dropout is disabled and all features are used (scaled by 0.7 to compensate). | |

### Results

| Metric | Value |
|--------|-------|
| Best val accuracy | **85.97%** (epoch 5/50) |
| Best val loss | 0.4606 |
| Best epoch | 5 |

### What the results tell us

- **5.04% → 85.97%**: a 17x improvement, entirely from using pretrained features. The model went from barely-better-than-random to correctly identifying the breed 86% of the time.
- **Best at epoch 5 out of 50**: the head learned the mapping very quickly. This makes sense — the backbone provides high-quality, stable features. The head just needs to learn a linear-ish mapping from 2048-d features to 120 classes, which is a relatively simple optimization problem.
- **Early convergence implies overfitting after epoch 5**: the head likely started memorizing training set patterns after this point. The training accuracy probably kept climbing while val accuracy stagnated or dropped.
- **85.97% with frozen backbone is strong**: this tells us that ImageNet features transfer very well to dog breeds (which makes sense — ImageNet includes ~120 dog breeds among its 1000 classes, so the features are already tuned for canine visual patterns).
- **~14% error rate remains**: the model still gets ~1 in 7 images wrong. These are likely visually similar breeds (e.g., Malinois vs German Shepherd, different terrier breeds). Reducing this error needs either better features (unfreezing the backbone) or better training (augmentation, regularization, scheduling).

### Comparison across iterations

| Metric | Baseline CNN | ResNet50 Frozen |
|--------|-------------|----------------|
| Val accuracy | 5.04% | **85.97%** |
| Val loss | 4.4487 | **0.4606** |
| Best epoch | 17/20 | **5/50** |
| Total params | ~25K | ~24.6M |
| Trainable params | ~25K | ~1.1M |
| Training time | ~15 min | ~30-45 min |

---

## Iteration 2: Unfreeze Backbone — First Attempt (Failed)

**Purpose**: unfreeze ResNet50's layer4 so the backbone can adapt its high-level features to dog breeds. The hypothesis was that generic ImageNet features aren't optimized for fine-grained breed differences (e.g., Malinois vs German Shepherd), and letting the last layer adapt would close that gap.

### What changed vs Iteration 1

| Parameter | Iter 1 (Frozen) | Iter 2 | Why |
|-----------|----------------|--------|-----|
| Frozen layers | All backbone | Layers 1-3 only | Layer4 has the most task-specific features. |
| Trainable params | 1.1M (head only) | 16.1M (layer4 + head) | 14.5x more params to train. |
| Backbone LR | — | 1e-4 | Differential LR: lower for pretrained weights. |
| Head LR | 1e-3 | 1e-3 | Same as before — randomly initialized. |
| Optimizer | Adam | AdamW | Decoupled weight decay for fine-tuning. |
| Weight decay | 0 | 1e-4 | Regularize the unfrozen backbone. |
| Epochs | 50 | 30 | Fine-tuning should converge faster. |

### Results

| Metric | Value |
|--------|-------|
| Best val accuracy | **85.13%** (epoch 27/30) |
| Best val loss | 0.8136 |
| Final train accuracy | 99.8% |
| Training time | ~64 min |

### Why it failed — overfitting

This iteration performed **worse** than the frozen backbone (85.13% vs 85.97%). The reason is clear from the training curves:

- **Train acc 99.8% vs val acc 85.1%** — a 15% gap. The model memorized the training set.
- **Val loss got worse over training**: 0.52 at epoch 2 → 0.81 at epoch 27. The model became more confident in its wrong answers.
- **Root cause**: 16.1M trainable params with only 8,177 training images (~2,000 params per image). The backbone LR of 1e-4 was too aggressive — it overwrote pretrained features faster than it could learn useful replacements (catastrophic forgetting).

### Lesson learned

Unfreezing more parameters doesn't automatically help. With a small dataset, the regularization must scale with the number of trainable parameters. Our dropout (0.3) and weight decay (1e-4) weren't enough to counteract 16M free parameters.

### Comparison across iterations

| Metric | Baseline CNN | ResNet50 Frozen | ResNet50 Fine-tune (v1) |
|--------|-------------|----------------|------------------------|
| Val accuracy | 5.04% | **85.97%** | 85.13% |
| Val loss | 4.4487 | **0.4606** | 0.8136 |
| Train-val gap | ~0% | ~10% | **~15%** |
| Trainable params | ~25K | ~1.1M | ~16.1M |

---

## Iteration 2b: Fine-tuning with Stronger Regularization

**Purpose**: same goal as Iteration 2 (adapt layer4 to dog breeds), but fix the overfitting with two targeted changes.

### What changes vs Iteration 2

| Parameter | Iter 2 (failed) | Iter 2b | Why |
|-----------|-----------------|---------|-----|
| Backbone LR | 1e-4 | **1e-5** | 10x more conservative. Preserves pretrained features while allowing gentle adaptation. |
| Dropout | 0.3 | **0.5** | Stronger regularization to match the larger number of trainable params. Drops 50% of features during training, forcing more redundant representations. |

Everything else stays the same (AdamW, weight_decay=1e-4, 30 epochs, same head LR 1e-3). Two changes, clear hypothesis: if overfitting was the problem, stronger regularization should fix it.

### Results

| Metric | Value |
|--------|-------|
| Best val accuracy | **87.48%** (epoch 4/30) |
| Best val loss | 0.4385 |
| Final train accuracy | 99.1% |
| Training time | ~66 min |

### What the results tell us

- **87.48% vs 85.97% (frozen)**: fine-tuning now actually helps (+1.5%). The lower backbone LR (1e-5 vs 1e-4) preserved pretrained features while allowing gentle adaptation.
- **Train-val gap reduced**: 15% (Iter 2) → ~9% (Iter 2b). Still overfitting, but much more controlled. Dropout 0.5 is doing its job.
- **Val loss stable**: 0.44 → 0.54 over 30 epochs. Compare to Iter 2 where val loss nearly doubled (0.52 → 0.81). The model isn't destroying its learned representations anymore.
- **Best at epoch 4**: the model converges fast, then slowly overfits. Most of the 30 epochs were unnecessary — but the best checkpoint was saved, so no harm done.
- **Remaining ~12.5% error**: the model still struggles with visually similar breeds. Further improvement likely requires either stronger augmentation (force the model to learn from harder examples) or better features (different architecture).

### Comparison across iterations

| Metric | Baseline CNN | ResNet50 Frozen | Fine-tune v1 | Fine-tune v2b |
|--------|-------------|----------------|--------------|---------------|
| Val accuracy | 5.04% | 85.97% | 85.13% | **87.48%** |
| Val loss | 4.4487 | 0.4606 | 0.8136 | **0.4385** |
| Train-val gap | ~0% | ~10% | ~15% | **~9%** |
| Trainable params | ~25K | ~1.1M | ~16.1M | ~16.1M |
| Best epoch | 17/20 | 5/50 | 27/30 | **4/30** |

---

## Iteration 3: Label Smoothing

**Purpose**: reduce overconfidence and improve generalization by softening the training targets. Instead of telling the model "this is 100% a golden retriever", we say "this is 99.17% a golden retriever and 0.07% each of the other 119 breeds."

### What is label smoothing?

Standard CrossEntropyLoss uses **hard targets**: the true class gets probability 1.0, everything else gets 0.0. This pushes the model to be infinitely confident — logits grow unbounded to make the softmax output closer to [0, 0, ..., 1, ..., 0]. This causes two problems:

1. **Overfitting**: the model wastes capacity memorizing "I'm absolutely certain this is breed X" instead of learning "breed X and Y look similar, but this one is probably X."
2. **Poor calibration**: the model outputs 99.9% confidence even when it's wrong. This makes the val loss worse because wrong-but-confident predictions are heavily penalized.

Label smoothing with `ε=0.1` redistributes 10% of the target probability uniformly across all classes:
- True class: `1 - ε + ε/120 = 0.9917`
- Other classes: `ε/120 = 0.00083`

The model can never reach 0 loss (even with perfect predictions), which prevents it from over-optimizing on the training set.

### What changed vs Iteration 2b

| Parameter | Iter 2b | Iter 3 | Why |
|-----------|---------|--------|-----|
| Loss function | `CrossEntropyLoss()` | `CrossEntropyLoss(label_smoothing=0.1)` | Prevents overconfident predictions, acts as implicit regularization. |

Everything else identical — same model, same LRs, same dropout, same epochs. One change, clean comparison.

### Results

| Metric | Value |
|--------|-------|
| Best val accuracy | **88.12%** (epoch 25/30) |
| Best val loss | 1.1951 (not comparable to previous — see note) |
| Final train accuracy | 99.5% |
| Training time | ~60 min |

**Note on loss values**: label smoothing raises the theoretical minimum loss from 0 to ~0.48 (the entropy of the smoothed distribution). So a val loss of 1.20 here is roughly equivalent to 0.72 without smoothing. Losses across Iter 2b and Iter 3 are not directly comparable.

### What the results tell us

- **87.48% → 88.12%** (+0.6%) from a one-line change. Small but meaningful — label smoothing isn't adding new information, it's just training more carefully.
- **Remarkably stable val accuracy**: the model never had a bad epoch. Val accuracy stayed in the 87-88% range from epoch 3 onward. Compare to Iter 2b where it peaked at epoch 4 and then drifted down.
- **Val loss flat**: 1.22 → 1.20 over 30 epochs. The model stopped overfitting — it reached its capacity and plateaued cleanly. This is exactly what label smoothing is designed to do.
- **Diminishing returns**: we went from 5% → 86% → 87.5% → 88.1%. Each step is smaller. We're approaching the ceiling of what this architecture + dataset combination can achieve without fundamentally different approaches.

### Comparison across all iterations

| Metric | Baseline CNN | ResNet50 Frozen | Fine-tune v1 | Fine-tune v2b | + Label Smoothing |
|--------|-------------|----------------|--------------|---------------|-------------------|
| Val accuracy | 5.04% | 85.97% | 85.13% | 87.48% | **88.12%** |
| Train-val gap | ~0% | ~10% | ~15% | ~9% | **~5%** |
| Trainable params | ~25K | ~1.1M | ~16.1M | ~16.1M | ~16.1M |
| Best epoch | 17/20 | 5/50 | 27/30 | 4/30 | **25/30** |
| Key change | — | Pretrained backbone | Unfreeze layer4 | Fix LR + dropout | Label smoothing |

---

## Analysis: Understanding the Best Model

This section analyzes the best model (Iteration 3 — fine-tuned ResNet50 with label smoothing, 88.12% val accuracy) to understand what it learned, where it fails, and why. All figures are in the `analysis/` directory.

### Top-K Accuracy

| K | Correct | Total | Accuracy |
|---|---------|-------|----------|
| Top-1 | 1,802 | 2,045 | **88.12%** |
| Top-3 | 1,995 | 2,045 | **97.56%** |
| Top-5 | 2,015 | 2,045 | **98.53%** |

**What this tells us**: when the model is wrong, the correct breed is almost always in its top-3 guesses (97.6%). Only 30 images out of 2,045 (1.5%) have the correct breed outside the top-5. This means the model has a strong understanding of visual similarity — it rarely makes completely unrelated predictions. Its errors are "near misses" between visually similar breeds, not fundamental failures.

### Most Confused Breed Pairs

*(see `analysis/confusion_matrix_top20.png`)*

| True breed | Predicted as | Rate |
|-----------|-------------|------|
| Eskimo dog | Siberian husky | 61.5% |
| Toy poodle | Miniature poodle | 43.8% |
| American staffordshire terrier | Staffordshire bullterrier | 33.3% |
| Collie | Border collie | 29.4% |
| Miniature poodle | Toy poodle | 31.2% |
| Cardigan | Pembroke | 26.7% |
| Lhasa | Shih-tzu | 22.2% |
| Walker hound | English foxhound | 28.6% |
| Wire-haired fox terrier | Lakeland terrier | 25.0% |

**Pattern**: every confused pair is a set of breeds that genuinely look alike. These aren't random errors — they're the exact pairs that even human experts find difficult:

- **Size-based distinctions** (toy vs miniature poodle) — impossible to determine from a single photo without a reference object.
- **Closely related breeds** (eskimo dog vs husky, cardigan vs pembroke corgi, collie vs border collie) — differ in subtle proportions, not major visual features.
- **Breed group lookalikes** (lhasa vs shih-tzu, walker hound vs english foxhound) — similar coat patterns and body structure, differ mainly in skull shape and size.

The confusion is also **symmetric** in most cases: eskimo dogs get predicted as huskies (61.5%) and huskies get predicted as eskimo dogs (21.1%). This confirms it's a feature similarity issue, not a class imbalance issue.

### Per-class Accuracy Distribution

*(see `analysis/per_class_accuracy_dist.png`)*

- **31 breeds at 100% accuracy** — the model perfectly classifies nearly a quarter of all breeds. These tend to be visually distinctive: papillon (butterfly ears), saluki (lean silhouette), whippet (unique build), yorkshire terrier (distinctive coat).
- **Most breeds cluster at 85-100%** — the distribution is heavily right-skewed. The model is reliably good across the board.
- **A few hard outliers**: eskimo dog (15.4%), walker hound (42.9%), toy poodle (50.0%). These are all breeds that are visually near-identical to a close relative.

### Grad-CAM: What the Model Looks At

*(see `analysis/gradcam_correct.png` and `analysis/gradcam_incorrect.png`)*

Grad-CAM produces heatmaps showing which regions of the image contributed most to the model's prediction. Hot regions (red/yellow) are the areas the model "focused on."

**Correct predictions**: the model consistently focuses on the **face, ears, muzzle, and body shape** — the same features a human would use. It ignores backgrounds, human hands/legs, and irrelevant objects. This confirms the model learned genuine breed features, not dataset shortcuts (like background correlations or watermarks).

**Incorrect predictions**: even when wrong, the model focuses on the right regions (face, body). The errors come from the features themselves being too similar between breeds (e.g., standard schnauzer vs giant schnauzer — same features, different scale). The model isn't "looking at the wrong thing" — it's looking at the right thing but the visual difference is too subtle to distinguish.

### Confidence Calibration

*(see `analysis/confidence_distribution.png`)*

| | Mean Confidence | Median Confidence |
|--|----------------|-------------------|
| Correct predictions (n=1,802) | 0.816 | 0.885 |
| Incorrect predictions (n=243) | 0.522 | 0.516 |

**The model knows when it's uncertain.** Correct predictions cluster at high confidence (0.85-0.95), while incorrect predictions spread flat around 0.50. This is excellent calibration, largely thanks to label smoothing — the model doesn't produce overconfident wrong answers.

This means in a production setting, you could set a confidence threshold (e.g., 0.7) and flag low-confidence predictions for human review, effectively boosting the system's reliability.

### Summary

The analysis reveals a model that:
1. **Understands dog breeds well** — 88% top-1, 98.5% top-5 accuracy.
2. **Fails only where expected** — confused pairs are breeds that genuinely look alike.
3. **Looks at the right features** — Grad-CAM confirms focus on face, ears, body, not shortcuts.
4. **Knows when it's uncertain** — well-calibrated confidence separates correct from incorrect predictions.

The remaining ~12% error rate is largely irreducible without either (a) more training data for the hardest breed pairs, or (b) a fundamentally different approach (e.g., fine-grained part-based models that explicitly detect ears, muzzle shape, coat texture separately).
