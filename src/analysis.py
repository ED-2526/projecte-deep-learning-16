"""
Analysis script for the best model checkpoint.

Produces:
1. Top-K accuracy (top-1, top-3, top-5)
2. Confusion matrix (top most-confused breed pairs)
3. Per-class accuracy (best and worst breeds)
4. Grad-CAM heatmaps (correct and incorrect predictions)
5. Error analysis (sample misclassified images)

All figures saved to analysis/ directory.

Usage:
    python -m src.analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from src.dataloaders.dataset import create_dataloaders
from src.models.transfer import FineTuneResNet50


# ─── Configuration ──────────────────────────────────────────────────────────

CHECKPOINT_PATH = "checkpoints/best_resnet50_finetune.pth"
OUTPUT_DIR = Path("analysis")
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


def load_model(checkpoint_path):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = FineTuneResNet50(num_classes=120).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    idx_to_breed = {i: breed for breed, i in ckpt["breed_to_idx"].items()}
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_accuracy']:.4f})")
    return model, idx_to_breed


@torch.no_grad()
def collect_predictions(model, val_loader):
    """Run inference on full val set, return all logits and true labels."""
    all_logits = []
    all_labels = []

    for images, labels in tqdm(val_loader, desc="Running inference"):
        images = images.to(DEVICE)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels)


# ─── 1. Top-K Accuracy ─────────────────────────────────────────────────────

def compute_topk_accuracy(logits, labels):
    """Compute top-1, top-3, top-5 accuracy."""
    results = {}
    for k in [1, 3, 5]:
        _, topk_preds = logits.topk(k, dim=1)
        correct = topk_preds.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
        results[k] = correct / len(labels)

    print("\n=== Top-K Accuracy ===")
    for k, acc in results.items():
        print(f"  Top-{k}: {acc:.4f} ({acc*100:.2f}%)")

    return results


# ─── 2. Per-class Accuracy ──────────────────────────────────────────────────

def compute_per_class_accuracy(logits, labels, idx_to_breed):
    """Compute accuracy per breed, return sorted."""
    preds = logits.argmax(dim=1)
    n_classes = len(idx_to_breed)

    per_class = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            acc = preds[mask].eq(c).float().mean().item()
            per_class[idx_to_breed[c]] = (acc, mask.sum().item())

    sorted_classes = sorted(per_class.items(), key=lambda x: x[1][0])

    print("\n=== Worst 10 Breeds ===")
    for breed, (acc, n) in sorted_classes[:10]:
        print(f"  {acc*100:5.1f}% ({n:3d} imgs) — {breed}")

    print("\n=== Best 10 Breeds ===")
    for breed, (acc, n) in sorted_classes[-10:]:
        print(f"  {acc*100:5.1f}% ({n:3d} imgs) — {breed}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    accs = [v[0] for v in per_class.values()]
    ax.hist(accs, bins=20, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Per-breed Accuracy")
    ax.set_ylabel("Number of Breeds")
    ax.set_title("Distribution of Per-breed Accuracy (Val Set)")
    ax.axvline(np.mean(accs), color="red", linestyle="--", label=f"Mean: {np.mean(accs):.2f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_class_accuracy_dist.png", dpi=150)
    plt.close()

    return per_class, sorted_classes


# ─── 3. Confusion Matrix ────────────────────────────────────────────────────

def plot_confusion_analysis(logits, labels, idx_to_breed):
    """Find and plot the most confused breed pairs."""
    preds = logits.argmax(dim=1)
    n_classes = len(idx_to_breed)

    # Build confusion matrix
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    # Find top confused pairs (off-diagonal)
    confused_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((
                    idx_to_breed[i], idx_to_breed[j],
                    cm[i, j].item(), cm[i].sum().item()
                ))

    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    print("\n=== Top 15 Most Confused Pairs ===")
    print(f"  {'True breed':<30} {'Predicted as':<30} {'Count':>5}  {'Rate':>6}")
    for true_b, pred_b, count, total in confused_pairs[:15]:
        print(f"  {true_b:<30} {pred_b:<30} {count:>5}  {count/total*100:>5.1f}%")

    # Plot top-20 most confused breeds as a sub-confusion matrix
    # Find the breeds involved in the most confusions
    breed_confusion_count = defaultdict(int)
    for true_b, pred_b, count, _ in confused_pairs:
        breed_confusion_count[true_b] += count
        breed_confusion_count[pred_b] += count

    top_confused_breeds = sorted(breed_confusion_count, key=breed_confusion_count.get, reverse=True)[:20]
    breed_to_idx_local = {b: i for i, b in enumerate(top_confused_breeds)}
    idx_to_global = {b: [k for k, v in idx_to_breed.items() if v == b][0] for b in top_confused_breeds}

    sub_cm = np.zeros((20, 20))
    for i, b1 in enumerate(top_confused_breeds):
        for j, b2 in enumerate(top_confused_breeds):
            sub_cm[i, j] = cm[idx_to_global[b1], idx_to_global[b2]].item()

    # Normalize by row (true class count)
    row_sums = sub_cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    sub_cm_norm = sub_cm / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        sub_cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=[b.replace("_", " ") for b in top_confused_breeds],
        yticklabels=[b.replace("_", " ") for b in top_confused_breeds],
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — 20 Most Confused Breeds (normalized by row)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_top20.png", dpi=150)
    plt.close()

    return confused_pairs


# ─── 4. Grad-CAM ────────────────────────────────────────────────────────────

class GradCAM:
    """Simple Grad-CAM implementation for ResNet50."""

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook into the last conv layer (layer4)
        target_layer = model.backbone.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image, class_idx=None):
        """Generate Grad-CAM heatmap for an image."""
        self.model.zero_grad()
        output = self.model(image)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass for the target class
        target = output[0, class_idx]
        target.backward()

        # Weight the activation maps by the mean gradient
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # global avg pool of gradients
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # only positive contributions

        # Resize to image size
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, output


def plot_gradcam_examples(model, val_loader, idx_to_breed):
    """Generate Grad-CAM heatmaps for a mix of correct and incorrect predictions."""
    gradcam = GradCAM(model)

    # Collect some correct and incorrect examples
    correct_examples = []
    incorrect_examples = []

    # Inverse normalization for display
    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])

    model.eval()
    for images, labels in val_loader:
        for i in range(images.size(0)):
            img = images[i:i+1].to(DEVICE)
            label = labels[i].item()

            cam, output = gradcam.generate(img)
            pred = output.argmax(dim=1).item()
            conf = F.softmax(output, dim=1)[0, pred].item()

            entry = {
                "image": inv_normalize(images[i]).clamp(0, 1).permute(1, 2, 0).numpy(),
                "cam": cam,
                "true": idx_to_breed[label],
                "pred": idx_to_breed[pred],
                "conf": conf,
            }

            if pred == label and len(correct_examples) < 6:
                correct_examples.append(entry)
            elif pred != label and len(incorrect_examples) < 6:
                incorrect_examples.append(entry)

            if len(correct_examples) >= 6 and len(incorrect_examples) >= 6:
                break
        if len(correct_examples) >= 6 and len(incorrect_examples) >= 6:
            break

    # Plot correct predictions
    _plot_gradcam_grid(correct_examples, "Grad-CAM — Correct Predictions", "gradcam_correct.png")
    # Plot incorrect predictions
    _plot_gradcam_grid(incorrect_examples, "Grad-CAM — Incorrect Predictions", "gradcam_incorrect.png")


def _plot_gradcam_grid(examples, title, filename):
    """Plot a 2x3 grid of Grad-CAM overlays."""
    n = len(examples)
    if n == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, ex in enumerate(examples[:6]):
        ax = axes[i]
        ax.imshow(ex["image"])
        ax.imshow(ex["cam"], cmap="jet", alpha=0.4)
        true_label = ex["true"].replace("_", " ")
        pred_label = ex["pred"].replace("_", " ")
        color = "green" if ex["true"] == ex["pred"] else "red"
        ax.set_title(f"True: {true_label}\nPred: {pred_label} ({ex['conf']*100:.1f}%)",
                     fontsize=9, color=color)
        ax.axis("off")

    # Hide unused axes
    for i in range(n, 6):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()


# ─── 5. Error Analysis Summary ──────────────────────────────────────────────

def error_summary(logits, labels, idx_to_breed):
    """Analyze the confidence distribution of correct vs incorrect predictions."""
    probs = F.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)
    confidences = probs.max(dim=1).values

    correct_mask = preds == labels
    correct_conf = confidences[correct_mask].numpy()
    incorrect_conf = confidences[~correct_mask].numpy()

    print("\n=== Confidence Analysis ===")
    print(f"  Correct predictions:   mean conf = {correct_conf.mean():.3f}, median = {np.median(correct_conf):.3f}")
    print(f"  Incorrect predictions: mean conf = {incorrect_conf.mean():.3f}, median = {np.median(incorrect_conf):.3f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(correct_conf, bins=30, alpha=0.6, label=f"Correct (n={len(correct_conf)})", color="green")
    ax.hist(incorrect_conf, bins=30, alpha=0.6, label=f"Incorrect (n={len(incorrect_conf)})", color="red")
    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confidence_distribution.png", dpi=150)
    plt.close()


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}")

    # Load model
    model, idx_to_breed = load_model(CHECKPOINT_PATH)

    # Load val data
    _, val_loader, _, _ = create_dataloaders(
        data_dir="data/", batch_size=32, image_size=224, val_split=0.2, num_workers=4, seed=42
    )

    # 1. Collect all predictions
    logits, labels = collect_predictions(model, val_loader)

    # 2. Top-K accuracy
    topk = compute_topk_accuracy(logits, labels)

    # 3. Per-class accuracy
    per_class, sorted_classes = compute_per_class_accuracy(logits, labels, idx_to_breed)

    # 4. Confusion matrix
    confused_pairs = plot_confusion_analysis(logits, labels, idx_to_breed)

    # 5. Confidence analysis
    error_summary(logits, labels, idx_to_breed)

    # 6. Grad-CAM (needs gradients, so separate pass)
    print("\n=== Generating Grad-CAM heatmaps ===")
    plot_gradcam_examples(model, val_loader, idx_to_breed)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
