#!/usr/bin/env python3
"""
Composite Image Augmentation for Attention Training
Creates 2x2 and 3x3 grids of images with combined labels to make model more robust
"""

import json
import random
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
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


def create_composite_image(images, grid_size=(2, 2)):
    """
    Create a composite image from multiple images in a grid

    Args:
        images: List of PIL Image objects
        grid_size: Tuple (rows, cols)

    Returns:
        Composite PIL Image
    """
    rows, cols = grid_size
    n_images = rows * cols

    if len(images) < n_images:
        raise ValueError(f"Need {n_images} images for {rows}x{cols} grid")

    # Get size (use first image as reference, resize all to same size)
    img_width, img_height = images[0].size

    # Resize all images to same size
    target_size = (img_width, img_height)
    resized_images = [img.resize(target_size, Image.Resampling.LANCZOS) for img in images[:n_images]]

    # Create composite
    composite_width = target_size[0] * cols
    composite_height = target_size[1] * rows
    composite = Image.new('RGB', (composite_width, composite_height))

    # Paste images
    for idx, img in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        x = col * target_size[0]
        y = row * target_size[1]
        composite.paste(img, (x, y))

    return composite


def combine_annotations(annotations_list, classes):
    """
    Combine annotations from multiple images

    Args:
        annotations_list: List of annotation dicts
        classes: List of class names

    Returns:
        Combined annotation dict
    """
    combined_counts = {cls: 0 for cls in classes}
    all_classes = set()

    for ann in annotations_list:
        counts = ann.get('counts', {})
        for cls in classes:
            combined_counts[cls] += counts.get(cls, 0)
            if counts.get(cls, 0) > 0:
                all_classes.add(cls)

    # Remove blank from classes present
    all_classes.discard('blank')

    # Find primary class (one with highest count)
    primary_class = None
    if all_classes:
        primary_class = max(all_classes, key=lambda c: combined_counts[c])

    return {
        'counts': combined_counts,
        'classes': list(all_classes),
        'primary_class': primary_class
    }


def generate_composite_dataset(annotations_file, output_dir, grid_sizes=[(2, 2), (3, 3)],
                               num_per_size=50, exclude_blank=True, allow_reuse=True):
    """
    Generate composite images from existing annotations

    Args:
        annotations_file: Path to annotations.json
        output_dir: Directory to save composite images
        grid_sizes: List of (rows, cols) tuples
        num_per_size: Number of composites to generate per grid size
        exclude_blank: Whether to exclude images labeled as "blank"
        allow_reuse: Allow same image to be used in multiple composites
    """
    # Load annotations
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    classes = data['classes']
    images_data = data['images']

    # Filter out blank-only images if requested
    if exclude_blank:
        valid_images = {}
        for img_name, img_data in images_data.items():
            counts = img_data.get('counts', {})
            # Check if there's at least one non-blank class with count > 0
            has_objects = any(counts.get(cls, 0) > 0 for cls in classes if cls != 'blank')
            if has_objects:
                valid_images[img_name] = img_data
    else:
        valid_images = images_data

    if len(valid_images) == 0:
        print("No valid images found for compositing!")
        print(f"\nDebug info:")
        print(f"  Total images in annotations: {len(images_data)}")
        print(f"  Classes: {classes}")
        if images_data:
            sample_name = list(images_data.keys())[0]
            sample_data = images_data[sample_name]
            print(f"  Sample image: {sample_name}")
            print(f"  Sample counts: {sample_data.get('counts', {})}")
        return

    print(f"Found {len(valid_images)} valid images for compositing")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # New annotations dict for composite images
    composite_annotations = {
        'classes': classes,
        'images': {}
    }

    # Load annotations file if it exists (to append)
    composite_ann_file = output_dir.parent / 'annotations' / 'annotations_composite.json'
    if composite_ann_file.exists():
        with open(composite_ann_file, 'r') as f:
            composite_annotations = json.load(f)

    image_names = list(valid_images.keys())

    # Generate composites for each grid size
    for rows, cols in grid_sizes:
        n_images = rows * cols

        print(f"\nGenerating {num_per_size} composite images of size {rows}x{cols}...")

        for i in tqdm(range(num_per_size), **get_tqdm_kwargs()):
            # Randomly sample images (with replacement if needed)
            if allow_reuse or len(image_names) >= n_images:
                sampled_names = random.choices(image_names, k=n_images)
            else:
                sampled_names = random.sample(image_names, n_images)
            sampled_data = [valid_images[name] for name in sampled_names]

            # Load images
            images = []
            for img_name in sampled_names:
                img_path = Path(valid_images[img_name]['path'])
                if not img_path.exists():
                    # Try relative to input directory
                    img_path = Path('input') / img_name
                if img_path.exists():
                    images.append(Image.open(img_path).convert('RGB'))

            if len(images) < n_images:
                print(f"Warning: Could not load enough images, skipping composite {i}")
                continue

            # Create composite image
            composite_img = create_composite_image(images, grid_size=(rows, cols))

            # Combine annotations
            combined_ann = combine_annotations(sampled_data, classes)

            # Save composite image
            composite_name = f"composite_{rows}x{cols}_{i:04d}.jpg"
            composite_path = output_dir / composite_name
            composite_img.save(composite_path, quality=95)

            # Add to annotations
            composite_annotations['images'][composite_name] = {
                'path': str(composite_path),
                'counts': combined_ann['counts'],
                'classes': combined_ann['classes'],
                'primary_class': combined_ann['primary_class'],
                'source_images': sampled_names,  # Track which images were combined
                'grid_size': f"{rows}x{cols}"
            }

    # Save composite annotations
    composite_ann_file.parent.mkdir(parents=True, exist_ok=True)
    with open(composite_ann_file, 'w') as f:
        json.dump(composite_annotations, f, indent=2)

    print(f"\n✓ Generated {len(composite_annotations['images'])} composite images")
    print(f"✓ Saved to: {output_dir}")
    print(f"✓ Annotations saved to: {composite_ann_file}")

    return composite_annotations


def merge_annotations(original_file, composite_file, output_file):
    """
    Merge original and composite annotations into a single file

    Args:
        original_file: Path to original annotations.json
        composite_file: Path to composite annotations.json
        output_file: Path to save merged annotations
    """
    with open(original_file, 'r') as f:
        original = json.load(f)

    with open(composite_file, 'r') as f:
        composite = json.load(f)

    # Merge
    merged = {
        'classes': original['classes'],
        'images': {**original['images'], **composite['images']}
    }

    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"\n✓ Merged {len(original['images'])} original + {len(composite['images'])} composite")
    print(f"✓ Total: {len(merged['images'])} images")
    print(f"✓ Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate composite images for robust training')
    parser.add_argument('--annotations', type=str, default='data/annotations/annotations.json',
                        help='Path to original annotations file')
    parser.add_argument('--output-dir', type=str, default='input/composites',
                        help='Directory to save composite images')
    parser.add_argument('--grid-sizes', type=str, default='2x2,3x3',
                        help='Grid sizes separated by comma (e.g., "2x2,3x3")')
    parser.add_argument('--num-per-size', type=int, default=50,
                        help='Number of composites to generate per grid size')
    parser.add_argument('--total-target', type=int, default=None,
                        help='Target total number of composite images (overrides num-per-size)')
    parser.add_argument('--exclude-blank', action='store_true', default=True,
                        help='Exclude images labeled as blank')
    parser.add_argument('--merge', action='store_true',
                        help='Merge composite annotations with original')
    parser.add_argument('--allow-reuse', action='store_true', default=True,
                        help='Allow same image to appear in multiple composites')
    args = parser.parse_args()

    # Parse grid sizes
    grid_sizes = []
    for size_str in args.grid_sizes.split(','):
        parts = size_str.strip().split('x')
        if len(parts) == 2:
            grid_sizes.append((int(parts[0]), int(parts[1])))

    # Calculate num_per_size if total_target is specified
    num_per_size = args.num_per_size
    if args.total_target:
        num_per_size = args.total_target // len(grid_sizes)

    print("=" * 60)
    print("🎨 Composite Image Generator")
    print("=" * 60)
    print(f"Original annotations: {args.annotations}")
    print(f"Output directory: {args.output_dir}")
    print(f"Grid sizes: {grid_sizes}")
    if args.total_target:
        print(f"Target total composites: {args.total_target}")
    print(f"Composites per size: {num_per_size}")
    print(f"Exclude blank: {args.exclude_blank}")
    print(f"Allow reuse: {args.allow_reuse}")
    print("=" * 60)

    # Generate composites
    composite_annotations = generate_composite_dataset(
        args.annotations,
        args.output_dir,
        grid_sizes=grid_sizes,
        num_per_size=num_per_size,
        exclude_blank=args.exclude_blank,
        allow_reuse=args.allow_reuse
    )

    # Merge if requested
    if args.merge and composite_annotations:
        composite_file = Path(args.output_dir).parent / 'annotations' / 'annotations_composite.json'
        merged_file = Path(args.annotations).parent / 'annotations_merged.json'
        merge_annotations(args.annotations, composite_file, merged_file)

        print("\n💡 To train with merged dataset, use:")
        print(f"   python train.py --annotations {merged_file}")


if __name__ == "__main__":
    main()
