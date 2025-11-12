#!/usr/bin/env python3
"""
Dynamic composite dataset that generates composites on-the-fly during training
No disk storage needed - saves space!
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from pathlib import Path
import numpy as np
import random
import time
from threading import Thread, Lock
from queue import Queue


class DynamicCompositeDataset(Dataset):
    """
    Dataset that mixes original images with dynamically generated composites
    Composites are created on-the-fly to save disk space
    """

    def __init__(self, annotations_file, transform=None, composite_ratio=0.7,
                 grid_sizes=[(2, 2), (3, 3)], prefetch_size=50):
        """
        Args:
            annotations_file: Path to annotations.json
            transform: Torchvision transforms
            composite_ratio: Fraction of dataset that should be composites (0.0-1.0)
            grid_sizes: List of (rows, cols) for composite grids
            prefetch_size: Number of composites to prefetch in background
        """
        with open(annotations_file, 'r') as f:
            self.data = json.load(f)

        self.classes = self.data["classes"]
        self.num_classes = len(self.classes)
        self.transform = transform
        self.grid_sizes = grid_sizes
        self.prefetch_size = prefetch_size

        # Filter original images (exclude blank-only)
        self.original_samples = []
        for img_name, img_data in self.data["images"].items():
            counts = img_data.get("counts", {})
            has_objects = any(counts.get(cls, 0) > 0 for cls in self.classes if cls != "blank")
            if has_objects:
                self.original_samples.append((Path(img_data["path"]), img_data))

        print(f"Loaded {len(self.original_samples)} original images")

        # Calculate dataset composition
        num_originals = len(self.original_samples)
        num_composites = int(num_originals * composite_ratio / (1 - composite_ratio))

        self.num_originals = num_originals
        self.num_composites = num_composites
        self.total_size = num_originals + num_composites

        print(f"Dataset composition:")
        print(f"  Original images: {num_originals}")
        print(f"  Dynamic composites: {num_composites}")
        print(f"  Total: {self.total_size}")
        print(f"  Composite ratio: {composite_ratio:.1%}")

        # Prefetch queue for background generation (disabled for now - multiprocessing conflict)
        # Will generate on-demand instead
        self.prefetch_enabled = False
        self.generation_times = []
        self.cache_hits = 0
        self.cache_misses = 0


    def _generate_composite(self):
        """Generate a single composite image with combined annotations"""
        # Random grid size
        rows, cols = random.choice(self.grid_sizes)
        n_images = rows * cols

        # Sample random images
        sampled = random.choices(self.original_samples, k=n_images)

        # Load images
        images = []
        annotations = []
        for img_path, img_data in sampled:
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                annotations.append(img_data)
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
                return None, None

        if len(images) < n_images:
            return None, None

        # Create composite image
        # Resize all to same size
        target_size = (224, 224)  # Base size before augmentation
        resized = [img.resize(target_size, Image.Resampling.LANCZOS) for img in images]

        # Create grid
        composite_width = target_size[0] * cols
        composite_height = target_size[1] * rows
        composite = Image.new('RGB', (composite_width, composite_height))

        for idx, img in enumerate(resized):
            row = idx // cols
            col = idx % cols
            x = col * target_size[0]
            y = row * target_size[1]
            composite.paste(img, (x, y))

        # Combine annotations
        combined_counts = {cls: 0 for cls in self.classes}
        for ann in annotations:
            counts = ann.get('counts', {})
            for cls in self.classes:
                combined_counts[cls] += counts.get(cls, 0)

        # Find primary class
        classes_present = [cls for cls in self.classes if combined_counts[cls] > 0 and cls != 'blank']
        if classes_present:
            primary_class = max(classes_present, key=lambda c: combined_counts[c])
        else:
            primary_class = self.classes[0] if self.classes else None

        combined_annotation = {
            'counts': combined_counts,
            'classes': classes_present,
            'primary_class': primary_class
        }

        return composite, combined_annotation

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        start_time = time.time()

        # First part of dataset: original images
        if idx < self.num_originals:
            img_path, annotation = self.original_samples[idx]
            image = Image.open(img_path).convert('RGB')

        # Second part: dynamic composites
        else:
            # Generate on-the-fly
            image, annotation = self._generate_composite()

            if image is None:
                # Fallback to random original image
                img_path, annotation = random.choice(self.original_samples)
                image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Create count vector
        count_vector = torch.zeros(self.num_classes, dtype=torch.float32)
        for i, class_name in enumerate(self.classes):
            count_vector[i] = annotation['counts'].get(class_name, 0)

        # Create class label
        primary_class = annotation.get('primary_class')
        if primary_class and primary_class in self.classes:
            class_idx = self.classes.index(primary_class)
        else:
            # Find first class with count > 0
            for i, class_name in enumerate(self.classes):
                if count_vector[i] > 0:
                    class_idx = i
                    break
            else:
                class_idx = 0

        # Track generation time
        gen_time = time.time() - start_time
        self.generation_times.append(gen_time)
        # Keep only last 100 times
        if len(self.generation_times) > 100:
            self.generation_times.pop(0)

        return image, class_idx, count_vector

    def get_stats(self):
        """Get statistics about composite generation"""
        if self.generation_times:
            avg_time = np.mean(self.generation_times)
            max_time = np.max(self.generation_times)
        else:
            avg_time = 0
            max_time = 0

        total_generated = len(self.generation_times)

        return {
            'avg_generation_time': avg_time,
            'max_generation_time': max_time,
            'total_generated': total_generated,
            'is_waiting': avg_time > 0.1  # >100ms avg = potential bottleneck
        }


def print_dataset_stats(dataset):
    """Print statistics about dynamic dataset"""
    if not isinstance(dataset, DynamicCompositeDataset):
        return

    stats = dataset.get_stats()

    print("\n" + "=" * 60)
    print("📊 Dynamic Composite Statistics")
    print("=" * 60)
    print(f"Total composites generated: {stats['total_generated']}")
    print(f"Avg generation time: {stats['avg_generation_time']*1000:.1f}ms")
    print(f"Max generation time: {stats['max_generation_time']*1000:.1f}ms")

    if stats['is_waiting']:
        print("⚠️  WARNING: Composite generation may be slow!")
        print("   Consider: reducing --composite-ratio or using fewer workers")
    else:
        print("✓ Composite generation is fast enough")

    print("=" * 60)
