# Model 1b: ResNet50 Frozen Backbone v2 — Deep Dive

## Relationship to v1

This is the **same model architecture** (ResNet50 frozen backbone + trainable head) with three targeted changes to training. Read [resnet50_deep_dive.md](resnet50_deep_dive.md) first — everything there still applies unless noted below.

**Unchanged:** backbone (ResNet50, frozen), optimizer (Adam), learning rate (1e-3), loss (CrossEntropyLoss), batch size (32), weight decay (0), head structure (2048→512→120), ImageNet normalization.

---

## What Changed and Why

### 1. Epochs: 15 → 50

**Why?**
- v1 ran for 15 epochs. At that point the model was still improving — we likely left accuracy on the table.
- 50 epochs gives the model time to fully converge and lets us observe the full learning curve: where it plateaus, when/if overfitting kicks in.
- Since we save the best checkpoint by val accuracy, extra epochs cost us only time, not accuracy.

**What to watch:**
- If val accuracy flatlines by epoch 20-25, the model has converged and we know 15 was close to enough.
- If val accuracy keeps climbing past epoch 30, we were definitely leaving performance behind.
- If val loss starts rising while train loss keeps dropping → classic overfitting signal.

### 2. LR Scheduler: None → CosineAnnealingLR

**Why?**
- With 50 epochs, a flat lr=1e-3 for the entire run is wasteful. Early on, the model needs large updates to move quickly. Late in training, smaller updates help it settle into a sharper minimum.
- CosineAnnealing smoothly decays the lr from 1e-3 → ~0 following a cosine curve over all 50 epochs.

**Why cosine and not StepLR or ReduceLROnPlateau?**
- **StepLR** (e.g., halve every 15 epochs): sharp drops create "jumps" in the loss curve. The model is cruising at one lr, then suddenly gets a smaller one. Works fine but cosine is smoother.
- **ReduceLROnPlateau**: reactive — waits for the model to plateau, then reduces. Good for exploration, but we already know our loss landscape is smooth (small head on fixed features). Cosine gives us a predictable, smooth schedule without needing to tune patience or factor.
- **CosineAnnealing** is the standard default for modern training (used in most recent papers) and needs only one hyperparameter: T_max (= total epochs).

**How it works in code:**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
# After each epoch:
scheduler.step()
```

The LR curve looks like:
```
lr
1e-3 ┤╲
     │  ╲
     │    ╲
     │      ╲
5e-4 ┤        ╲
     │          ╲
     │            ╲
     │              ╲
~0   ┤                ╲___
     └──────────────────────
     0    12   25   37   50  epoch
```

### 3. Dropout: 0.5 → 0.3

**Why?**
- v1's 0.5 dropout was the "safe" choice — maximum regularization per Srivastava et al. (2014).
- But with a frozen backbone, the head is already constrained: it only has 1.1M trainable params and the features it receives don't change. This is already a form of implicit regularization.
- 0.5 means half the head's features are zeroed every forward pass — the model may be losing too much useful signal, especially for fine-grained distinctions (e.g., similar breeds).
- 0.3 still regularizes (30% of features zeroed) but lets more information flow through, giving the head a better shot at learning subtle breed differences.

**What to watch:**
- If train accuracy >> val accuracy (big gap), we under-regularized and 0.5 was better.
- If the gap stays similar to v1 but both accuracies are higher, 0.3 was the right call.

---

## Changes Summary Table

| Parameter | v1 | v2 | Why the change |
|-----------|----|----|---------------|
| Epochs | 15 | **50** | See full convergence curve |
| LR Scheduler | None | **CosineAnnealingLR** | Smooth lr decay prevents late-training oscillation |
| Dropout | 0.5 | **0.3** | Less aggressive — let more signal through for fine-grained task |
| Everything else | — | Same | Isolate impact of these 3 changes |

---

## What to Expect

| Metric | v1 (15 epochs) | v2 (50 epochs, expected) |
|--------|----------------|--------------------------|
| Val accuracy | (check v1 run) | +2-5% over v1 |
| Convergence epoch | ~10-12 | ~25-35 |
| Training time | ~10-15 min | ~30-45 min |

The combination of more epochs + cosine LR + gentler dropout should push accuracy up modestly. The main value of this run is **understanding the full learning dynamics** — seeing the complete convergence curve tells us whether the frozen approach has headroom left or if we need to unfreeze the backbone next.
