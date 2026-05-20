# Dog Breed Identification — Deep Learning Project

**Course:** Deep Learning, UAB (4th year Data Engineering)
**Dataset:** [Kaggle Dog Breed Identification](https://www.kaggle.com/competitions/dog-breed-identification/data) — 120 breeds, ~10k training images
**Starting Point:** [Bycarkos/Starting-Points](https://github.com/Bycarkos/Starting-Points/tree/main/Dog-Classification)

---

## Phase 0 — Project Setup
- [x] Create roadmap
- [x] Set up repo structure (src, notebooks, configs, data/)
- [x] Create `CLAUDE.md` with project conventions
- [x] Port professor's starting point code into src/
- [x] Create .gitignore
- [x] Set up environment (venv + install dependencies)

## Phase 1 — Data Pipeline
- [x] Download Kaggle dataset (10,222 train / 10,357 test / 120 breeds)
- [x] Explore data: breed distribution, image sizes, sample visualization (EDA notebook)
- [x] Implement `dataset.py`: image loading (PIL), transforms (resize, normalize, augmentation)
- [x] Train/validation split (stratified by breed) — 8,177 train / 2,045 val
- [x] DataLoaders with batching (32 images/batch, 224x224px)

## Phase 2 — Baseline Model
- [x] Port professor's baseline CNN into the project
- [x] Training loop (loss, optimizer, metrics)
- [x] Train baseline, log accuracy & loss curves (wandb)
- [x] Establish benchmark to beat

### Baseline Results
| Metric | Value |
|--------|-------|
| Best val accuracy | **5.04%** (epoch 17/20) |
| Best val loss | 4.4487 |
| Model params | ~60K |
| Random chance | 0.83% (1/120) |

The baseline learns something (5% vs 0.8% random), but is far too shallow and small
to distinguish 120 visually similar breeds. This is the benchmark to beat in Phase 3.

## Phase 3 — Model Improvements
- [ ] Transfer learning — fine-tune pretrained model (ResNet50, EfficientNet, etc.)
- [ ] Data augmentation — RandomCrop, ColorJitter, horizontal flip, mixup
- [ ] Learning rate scheduling — CosineAnnealing, ReduceLROnPlateau
- [ ] Regularization — Dropout, weight decay, label smoothing
- [ ] Compare architectures systematically

## Phase 4 — Analysis & Experimentation
- [ ] Confusion matrix — which breeds get confused?
- [ ] Grad-CAM / attention visualization
- [ ] Ablation studies — what contributes most?
- [ ] Error analysis on misclassified images

## Phase 5 — Report & Presentation
- [ ] Write report: problem, methodology, experiments, results, conclusions
- [ ] Prepare presentation slides
- [ ] Clean up code and notebooks
