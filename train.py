#!/usr/bin/env python3
"""
Training script for weakly-supervised object detection with counting
Uses ResNet50 backbone with classification + density heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import os
import sys


def is_tmux():
    """Detect if running inside tmux"""
    return os.environ.get('TMUX') is not None


def get_tqdm_kwargs():
    """Get tqdm configuration optimized for current environment"""
    if is_tmux():
        # Tmux-friendly settings: less frequent updates, no dynamic sizing
        return {
            'file': sys.stdout,
            'dynamic_ncols': False,
            'ncols': 100,
            'position': 0,
            'leave': True,
            'mininterval': 2.0,  # Update every 2 seconds
            'maxinterval': 4.0,  # Max interval between updates
            'miniters': 5,  # Minimum iterations between updates
            'ascii': True,  # Use ASCII characters only
            'bar_format': '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        }
    else:
        # Regular terminal settings
        return {
            'file': sys.stdout,
            'dynamic_ncols': True,
            'position': 0,
            'leave': True
        }


class WeakCountModel(nn.Module):
    """
    Weakly-supervised counting model with:
    - Classification head (for class presence)
    - Density head (for counting and localization)
    """

    def __init__(self, num_classes):
        super().__init__()
        # Backbone without avgpool/fc
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4
        )
        ch = 2048

        # Classification branch (for stable learning + presence/absence)
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(ch, num_classes)

        # Density branch (counting map per class, non-negative)
        self.den_head = nn.Sequential(
            nn.Conv2d(ch, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, 1),
            nn.Softplus()  # >= 0, more stable than ReLU for small counts
        )

    def forward(self, x):
        feats = self.stem(x)  # (B, 2048, H/32, W/32)

        # Classification (for presence/absence)
        g = self.cls_pool(feats).flatten(1)  # (B, 2048)
        logits = self.cls_head(g)  # (B, C)

        # Density + counting
        den = self.den_head(feats)  # (B, C, h, w)
        counts = den.flatten(2).sum(-1)  # (B, C) global sum = count

        return logits, den, counts


class AnnotatedImageDataset(Dataset):
    """Dataset for loading annotated images"""

    def __init__(self, annotations_file, transform=None):
        with open(annotations_file, 'r') as f:
            self.data = json.load(f)

        self.classes = self.data["classes"]
        self.num_classes = len(self.classes)

        # Filter images that have annotations (excluding "blank" class)
        self.samples = []
        for img_name, img_data in self.data["images"].items():
            counts = img_data.get("counts", {})
            # Check if any non-blank class has count > 0
            has_objects = any(count > 0 for cls, count in counts.items() if cls != "blank")
            if has_objects:
                self.samples.append((Path(img_data["path"]), img_data))

        self.transform = transform
        print(f"Loaded {len(self.samples)} annotated images with {self.num_classes} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, annotation = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Create count vector
        count_vector = torch.zeros(self.num_classes, dtype=torch.float32)
        for i, class_name in enumerate(self.classes):
            count_vector[i] = annotation["counts"].get(class_name, 0)

        # Create class label (primary class)
        primary_class = annotation.get("primary_class")
        if primary_class and primary_class in self.classes:
            class_idx = self.classes.index(primary_class)
        else:
            # Find first class with count > 0
            for i, class_name in enumerate(self.classes):
                if count_vector[i] > 0:
                    class_idx = i
                    break
            else:
                class_idx = 0  # Default to first class if all zeros

        return image, class_idx, count_vector


def tv_loss(x):
    """Total variation loss for smoothness"""
    return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + \
           (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()


def poisson_loss(pred_counts, true_counts):
    """Poisson NLL loss for counting"""
    return F.poisson_nll_loss(pred_counts, true_counts, log_input=False, full=True)


def improved_count_loss(pred_counts, true_counts, alpha=2.0):
    """
    Improved counting loss that:
    - Heavily penalizes false positives (predicting objects when there are none)
    - Uses squared error to punish larger errors more
    - Gives bonus for exact matches

    Args:
        pred_counts: Predicted counts (B, C)
        true_counts: True counts (B, C)
        alpha: Weight for false positive penalty
    """
    # Squared L2 loss (punishes larger errors more than L1)
    base_loss = F.mse_loss(pred_counts, true_counts, reduction='none')

    # False positive penalty: predicting > 0 when truth is 0
    false_positive_mask = (true_counts == 0) & (pred_counts > 0.5)
    fp_penalty = alpha * (pred_counts ** 2) * false_positive_mask.float()

    # Combine
    total_loss = base_loss + fp_penalty

    return total_loss.mean()


def train_epoch(model, dataloader, optimizer, device, lambda_count=1.5, lambda_l1=1e-5, lambda_tv=1e-5, use_improved_loss=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_cnt_loss = 0

    ce = nn.CrossEntropyLoss()

    pbar = tqdm(dataloader, desc="Training", **get_tqdm_kwargs())
    for batch_idx, (x, y_cls, y_count) in enumerate(pbar):
        x = x.to(device)
        y_cls = y_cls.to(device)
        y_count = y_count.to(device)

        # Forward pass
        logits, den, counts = model(x)

        # Losses
        loss_cls = ce(logits, y_cls)

        # Use improved count loss that penalizes false positives more
        if use_improved_loss:
            loss_cnt = improved_count_loss(counts, y_count, alpha=2.0)
        else:
            loss_cnt = poisson_loss(counts, y_count)

        loss_reg = lambda_l1 * den.mean() + lambda_tv * tv_loss(den)

        loss = loss_cls + lambda_count * loss_cnt + loss_reg

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_cnt_loss += loss_cnt.item()

        # Update progress bar
        if is_tmux():
            # Simple print updates for tmux (every 10 batches)
            if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                progress = (batch_idx + 1) / len(dataloader) * 100
                print(f"\rTraining: {progress:3.0f}% ({batch_idx+1}/{len(dataloader)}) - "
                      f"loss={loss.item():.4f}, cls={loss_cls.item():.4f}, cnt={loss_cnt.item():.4f}",
                      end='', flush=True)
        else:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{loss_cls.item():.4f}',
                'cnt': f'{loss_cnt.item():.4f}'
            })

    if is_tmux():
        print()  # New line after training loop
    pbar.close()

    n = len(dataloader)
    return total_loss / n, total_cls_loss / n, total_cnt_loss / n


def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    correct = 0
    total = 0
    count_mae = 0
    count_mse = 0
    exact_matches = 0
    count_samples = 0

    ce = nn.CrossEntropyLoss()
    total_loss = 0
    total_cls_loss = 0
    total_cnt_loss = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", **get_tqdm_kwargs())
        for batch_idx, (x, y_cls, y_count) in enumerate(pbar):
            x = x.to(device)
            y_cls = y_cls.to(device)
            y_count = y_count.to(device)

            logits, den, counts = model(x)

            # Classification accuracy
            pred_cls = logits.argmax(1)
            correct += (pred_cls == y_cls).sum().item()
            total += x.size(0)

            # Count metrics
            count_diff = (counts - y_count).abs()
            count_mae += count_diff.sum().item()
            count_mse += (count_diff ** 2).sum().item()

            # Exact count matches (rounded predictions)
            rounded_preds = counts.round()
            exact_matches += (rounded_preds == y_count).all(dim=1).sum().item()

            count_samples += y_count.numel()

            # Losses
            loss_cls = ce(logits, y_cls)
            loss_cnt = improved_count_loss(counts, y_count, alpha=2.0)
            total_cls_loss += loss_cls.item()
            total_cnt_loss += loss_cnt.item()
            total_loss += (loss_cls + loss_cnt).item()

            # Update progress for tmux
            if is_tmux() and (batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1):
                progress = (batch_idx + 1) / len(dataloader) * 100
                print(f"\rValidation: {progress:3.0f}% ({batch_idx+1}/{len(dataloader)})",
                      end='', flush=True)

        if is_tmux():
            print()  # New line after validation loop
        pbar.close()

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'cls_loss': total_cls_loss / n,
        'cnt_loss': total_cnt_loss / n,
        'accuracy': correct / total,
        'count_mae': count_mae / count_samples,
        'count_rmse': (count_mse / count_samples) ** 0.5,
        'exact_count_acc': exact_matches / total
    }


def main():
    parser = argparse.ArgumentParser(description='Train weakly-supervised counting model')
    parser.add_argument('--annotations', type=str, default='data/annotations/annotations.json',
                        help='Path to annotations file')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--output', type=str, default='output/model.pth', help='Output model path')
    parser.add_argument('--input-size', type=int, default=448, help='Input image size')
    parser.add_argument('--use-dynamic', action='store_true',
                        help='Use dynamic composite generation (no disk storage)')
    parser.add_argument('--composite-ratio', type=float, default=0.7,
                        help='Ratio of composites in dynamic dataset (0.0-1.0)')
    parser.add_argument('--prefetch-size', type=int, default=50,
                        help='Number of composites to prefetch in background')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume training from saved checkpoint (default: True)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Start training from scratch (ignore existing checkpoint)')
    parser.add_argument('--reset-best', action='store_true',
                        help='Reset best score and re-evaluate after N epochs (useful after changing loss function)')
    parser.add_argument('--reset-eval-epochs', type=int, default=3,
                        help='Number of epochs to train before re-evaluating and setting new best (default: 3)')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check environment
    if is_tmux():
        print("📟 Running in tmux - using tmux-optimized progress bars")
    else:
        print("💻 Running in standard terminal")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize(int(args.input_size * 1.1)),
        transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Validation transform with padding (no cropping to preserve all objects)
    class PadToSquare:
        """Pad image to square while maintaining aspect ratio"""
        def __init__(self, target_size=448):
            self.target_size = target_size

        def __call__(self, img):
            w, h = img.size
            scale = self.target_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Pad to square
            img_padded = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
            paste_x = (self.target_size - new_w) // 2
            paste_y = (self.target_size - new_h) // 2
            img_padded.paste(img_resized, (paste_x, paste_y))
            return img_padded

    val_transform = transforms.Compose([
        PadToSquare(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    if args.use_dynamic:
        print("\n🚀 Using dynamic composite generation (no disk storage)")
        from dynamic_dataset import DynamicCompositeDataset

        # Create full dataset with dynamic composites
        full_dataset = DynamicCompositeDataset(
            args.annotations,
            transform=train_transform,
            composite_ratio=args.composite_ratio,
            grid_sizes=[(2, 2), (3, 3)],
            prefetch_size=args.prefetch_size
        )

        # Split into train/val
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Val dataset uses same dynamic dataset (will use originals mostly)
        # But we should use val_transform
        # Note: Transforms already applied in DynamicCompositeDataset
    else:
        # Traditional static dataset
        full_dataset = AnnotatedImageDataset(args.annotations, transform=train_transform)

        # Split into train/val
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Update val_dataset transform
        val_dataset.dataset.transform = val_transform

    # Dataloaders (use fewer workers to avoid "too many open files" on macOS)
    num_workers = 2 if args.use_dynamic else 4  # Dynamic generation needs fewer workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    num_classes = len(full_dataset.classes)
    model = WeakCountModel(num_classes).to(device)
    print(f"Model created with {num_classes} classes: {full_dataset.classes}")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0
    start_epoch = 0
    best_model_path = args.output

    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if requested
    reset_best_after_epochs = None
    if args.resume and Path(best_model_path).exists():
        print(f"\n📂 Loading checkpoint from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if args.reset_best:
            # Reset best score, will re-evaluate after N epochs
            best_val_acc = 0
            start_epoch = checkpoint.get('epoch', 0) + 1
            reset_best_after_epochs = start_epoch + args.reset_eval_epochs
            print(f"✓ Resumed from epoch {start_epoch}")
            print(f"⚠️  Best score RESET - will re-evaluate after epoch {reset_best_after_epochs}")
        else:
            # Use existing best score (old metric format compatibility)
            old_best = checkpoint.get('combined_score', checkpoint.get('val_accuracy', 0))
            best_val_acc = old_best
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"✓ Resumed from epoch {start_epoch}, best score: {best_val_acc:.3f}")
    elif args.resume:
        print(f"⚠️  No checkpoint found at {best_model_path}, starting from scratch")

    # Print initial stats if using dynamic dataset
    if args.use_dynamic and hasattr(train_dataset.dataset, 'get_stats'):
        from dynamic_dataset import print_dataset_stats
        print_dataset_stats(train_dataset.dataset)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_cls_loss, train_cnt_loss = train_epoch(
            model, train_loader, optimizer, device
        )

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        # Print metrics
        print(f"Train - Loss: {train_loss:.4f}, Cls: {train_cls_loss:.4f}, Cnt: {train_cnt_loss:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.3f}, "
              f"MAE: {val_metrics['count_mae']:.3f}, "
              f"RMSE: {val_metrics['count_rmse']:.3f}, "
              f"Exact: {val_metrics['exact_count_acc']:.3f}")

        # Print dynamic dataset stats every 5 epochs
        if args.use_dynamic and epoch % 5 == 0 and hasattr(train_dataset.dataset, 'get_stats'):
            from dynamic_dataset import print_dataset_stats
            print_dataset_stats(train_dataset.dataset)

        # Save best model (prioritize exact count accuracy over classification)
        current_score = val_metrics['exact_count_acc'] * 0.6 + val_metrics['accuracy'] * 0.4
        best_score = best_val_acc if isinstance(best_val_acc, float) else 0

        # Force save after reset_best_after_epochs to establish new baseline
        force_save = reset_best_after_epochs is not None and epoch >= reset_best_after_epochs

        if current_score > best_score or force_save:
            if force_save and reset_best_after_epochs == epoch:
                print(f"📊 Re-evaluation complete! Setting new baseline score: {current_score:.3f}")
                reset_best_after_epochs = None  # Only force once

            best_val_acc = current_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'val_mae': val_metrics['count_mae'],
                'val_exact_count': val_metrics['exact_count_acc'],
                'combined_score': current_score,
                'classes': full_dataset.classes,
                'num_classes': num_classes,
            }, best_model_path)
            print(f"✓ Saved best model (score: {best_val_acc:.3f}, exact: {val_metrics['exact_count_acc']:.3f})")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
