# Dog Breed Identification — Deep Learning Project (UAB)

## Context
- 4th year Data Engineering student at Universitat Autonoma de Barcelona
- Deep Learning course project
- Task: classify 120 dog breeds using the Kaggle Dog Breed Identification dataset
- Professor's starting point: simple 2-layer CNN baseline + dataset stub

## Project Structure
```
src/
  models/       # Model architectures (baseline.py, future improved models)
  dataloaders/  # Dataset classes and data loading utilities
  utils/        # Training helpers, metrics, visualization
notebooks/      # Exploration, EDA, experiment notebooks
configs/        # Training configs (hyperparams, paths)
data/           # Dataset files (gitignored)
```

## Conventions
- Framework: PyTorch + torchvision
- Python 3.10+
- Images loaded via PIL, transforms via torchvision.transforms
- All models inherit from nn.Module
- Training configs kept separate from code (in configs/ or as args)
- Keep notebooks for exploration, .py files for reusable code

## Current Phase
Phase 0 — Project Setup (see ROADMAP.md for full plan)

## Dataset
- Source: https://www.kaggle.com/competitions/dog-breed-identification/data
- 120 breeds, ~10k labeled training images, ~10k test images
- Labels in labels.csv (id → breed mapping)
- Images in train/ and test/ directories (JPEG)
