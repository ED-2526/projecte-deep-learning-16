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
- [ ] Set up environment (install dependencies)

## Phase 1 — Data Pipeline
- [ ] Download Kaggle dataset
- [ ] Explore data: breed distribution, image sizes, sample visualization
- [ ] Implement `dataset.py`: image loading (PIL), transforms (resize, normalize, augmentation)
- [ ] Train/validation split (stratified by breed)
- [ ] DataLoaders with batching

## Phase 2 — Baseline Model
- [ ] Port professor's baseline CNN into the project
- [ ] Training loop (loss, optimizer, metrics)
- [ ] Train baseline, log accuracy & loss curves
- [ ] Establish benchmark to beat

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
