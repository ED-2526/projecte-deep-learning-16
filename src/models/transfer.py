import torch
import torch.nn as nn
from torchvision import models


class TransferResNet50(nn.Module):
    """
    ResNet50 with frozen backbone + trainable classifier head.

    The backbone (all convolutional layers) is pretrained on ImageNet and frozen.
    Only the classifier head is trained from scratch on our dog breed dataset.

    Architecture:
        ResNet50 backbone (frozen, 23.5M params) → feature vector (2048-d)
        → Linear(2048, 512) → ReLU → Dropout(0.5) → Linear(512, num_classes)

    Args:
        num_classes: number of output classes (120 for dog breeds)
    """

    def __init__(self, num_classes):
        super(TransferResNet50, self).__init__()

        # Load ResNet50 pretrained on ImageNet
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Freeze all backbone parameters — no gradients, no updates
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace the original classifier (Linear(2048, 1000)) with our head
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def trainable_params(self):
        """Returns only the parameters that will be updated during training."""
        return [p for p in self.parameters() if p.requires_grad]


class FineTuneResNet50(nn.Module):
    """
    ResNet50 with partially unfrozen backbone for fine-tuning.

    Freezes early layers (conv1, bn1, layer1, layer2, layer3) which encode
    general visual features (edges, textures, shapes). Unfreezes layer4 which
    encodes high-level, task-specific features that benefit from adaptation.

    Uses differential learning rates:
    - Backbone layer4: lower LR (e.g. 1e-4) to preserve pretrained knowledge
    - Classifier head: higher LR (e.g. 1e-3) for faster learning from scratch

    Architecture:
        ResNet50 backbone (layer4 unfrozen) → feature vector (2048-d)
        → Linear(2048, 512) → ReLU → Dropout(0.5) → Linear(512, num_classes)
    """

    def __init__(self, num_classes):
        super(FineTuneResNet50, self).__init__()

        # Load ResNet50 pretrained on ImageNet
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze layer4 — the deepest residual block (most task-specific)
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        # Replace classifier head (stronger dropout than frozen version to combat overfitting)
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def get_param_groups(self, lr_backbone=1e-4, lr_head=1e-3):
        """
        Returns parameter groups with differential learning rates.

        - layer4 params: lr_backbone (small updates to preserve pretrained knowledge)
        - head params: lr_head (larger updates for randomly initialized weights)
        """
        backbone_params = list(self.backbone.layer4.parameters())
        head_params = list(self.backbone.fc.parameters())

        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ]


if __name__ == "__main__":
    model = TransferResNet50(num_classes=120)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"Total params:     {total:>12,}")
    print(f"Frozen (backbone): {frozen:>11,}")
    print(f"Trainable (head):  {trainable:>11,}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
