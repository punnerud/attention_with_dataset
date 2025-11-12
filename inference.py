#!/usr/bin/env python3
"""
Inference script for visualizing attention/density maps
Shows where the model thinks objects are located
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import cv2

from train import WeakCountModel


def load_model(model_path, device):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device)

    num_classes = checkpoint['num_classes']
    classes = checkpoint['classes']

    model = WeakCountModel(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Classes: {classes}")
    print(f"Validation accuracy: {checkpoint.get('val_accuracy', 'N/A')}")

    return model, classes


def preprocess_image(image_path, input_size=448):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.1)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor


def denormalize_image(tensor):
    """Denormalize image tensor for display"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img


def upsample_density_map(dmap, target_size):
    """Upsample density map to target size"""
    # dmap: (h, w) -> (H, W)
    dmap_normalized = (dmap - dmap.min()) / (dmap.max() - dmap.min() + 1e-8)
    return cv2.resize(dmap_normalized, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)


def find_peaks(dmap, k=None, thr=0.3):
    """
    Find peaks in density map
    dmap: (h,w) density map
    k: desired number of instances (predicted count rounded)
    thr: relative threshold vs dmap.max()
    return: list of (y,x) peak points
    """
    H, W = dmap.shape
    x = torch.from_numpy(dmap).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # Non-maximum suppression with maxpool
    nms = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)
    mask = (x == nms) & (x >= thr * x.max())

    ys, xs = mask[0, 0].nonzero(as_tuple=True)
    pts = [(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())]

    # Sort by score
    scores = x[0, 0, ys, xs]
    idx = scores.argsort(descending=True).tolist()
    pts = [pts[i] for i in idx]

    return pts[:k] if k else pts


def visualize_attention(image, density_maps, counts, classes, peaks_list, save_path=None):
    """
    Visualize attention overlays for all classes in 3x2 grid
    """
    num_classes = len(classes)
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    axes = axes.flatten()

    if num_classes == 1:
        axes = axes.reshape(-1, 1)

    for i, class_name in enumerate(classes):
        # Overlay subplot (even indices: 0, 2, 4)
        ax_overlay = axes[i * 2]
        ax_overlay.imshow(image)
        dmap = density_maps[i]

        if dmap.max() > 0:
            ax_overlay.imshow(dmap, cmap='jet', alpha=0.5, vmin=0, vmax=dmap.max())

            # Draw peaks
            peaks = peaks_list[i]
            if peaks:
                ys, xs = zip(*peaks)
                ax_overlay.scatter(xs, ys, c='white', s=200, marker='x', linewidths=3)
                ax_overlay.scatter(xs, ys, c='red', s=100, marker='x', linewidths=2)

        count = counts[i]
        ax_overlay.set_title(f'{class_name}\nCount: {count:.1f} (detected: {len(peaks_list[i])})', fontsize=12)
        ax_overlay.axis('off')

        # Density map subplot (odd indices: 1, 3, 5)
        ax_density = axes[i * 2 + 1]
        if dmap.max() > 0:
            im = ax_density.imshow(dmap, cmap='jet', vmin=0, vmax=dmap.max())
            plt.colorbar(im, ax=ax_density, fraction=0.046, pad=0.04)
        else:
            ax_density.imshow(np.zeros_like(dmap), cmap='gray')
        ax_density.set_title(f'Density Map', fontsize=12)
        ax_density.axis('off')

    plt.tight_layout(pad=1.0, h_pad=1.5, w_pad=1.0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def inference_single_image(model, classes, image_path, device, input_size=448, save_dir=None, peak_threshold=0.3):
    """Run inference on a single image"""
    # Load and preprocess
    img_original, img_tensor = preprocess_image(image_path, input_size)
    img_tensor = img_tensor.to(device)

    # Inference
    with torch.no_grad():
        logits, den, counts = model(img_tensor)

    # Get predictions
    pred_class_idx = logits.argmax(1).item()
    pred_class = classes[pred_class_idx]

    # Process density maps
    den = den[0].cpu().numpy()  # (C, h, w)
    counts = counts[0].cpu().numpy()  # (C,)

    # Original image size for upsampling
    orig_size = img_original.size  # (W, H)

    # Upsample density maps and find peaks
    density_maps = []
    peaks_list = []

    for c in range(len(classes)):
        dmap = den[c]
        count = float(counts[c])
        k = max(0, int(round(count)))

        # Upsample to original size
        dmap_up = upsample_density_map(dmap, (orig_size[1], orig_size[0]))

        # Find peaks
        peaks = find_peaks(dmap_up, k=k, thr=peak_threshold)

        density_maps.append(dmap_up)
        peaks_list.append(peaks)

    # Print results
    print(f"\nImage: {image_path}")
    print(f"Primary prediction: {pred_class}")
    print("\nCounts per class:")
    for i, class_name in enumerate(classes):
        print(f"  {class_name}: {counts[i]:.2f} (peaks detected: {len(peaks_list[i])})")

    # Visualize
    if save_dir:
        save_path = Path(save_dir) / f"{Path(image_path).stem}_attention.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        save_path = None

    visualize_attention(img_original, density_maps, counts, classes, peaks_list, save_path)

    return {
        'pred_class': pred_class,
        'counts': counts,
        'density_maps': density_maps,
        'peaks': peaks_list
    }


def inference_directory(model, classes, image_dir, device, input_size=448, save_dir=None, peak_threshold=0.3):
    """Run inference on all images in a directory"""
    image_dir = Path(image_dir)
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    ])

    print(f"Found {len(image_files)} images in {image_dir}")

    results = {}
    for image_path in image_files:
        try:
            result = inference_single_image(
                model, classes, image_path, device,
                input_size, save_dir, peak_threshold
            )
            results[image_path.name] = result
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run inference and visualize attention')
    parser.add_argument('--model', type=str, default='output/model.pth',
                        help='Path to trained model')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image-dir', type=str, default='input',
                        help='Path to directory of images')
    parser.add_argument('--save-dir', type=str, default='output/visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--input-size', type=int, default=448,
                        help='Input image size (should match training)')
    parser.add_argument('--peak-threshold', type=float, default=0.3,
                        help='Threshold for peak detection (0-1)')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, classes = load_model(args.model, device)

    # Run inference
    if args.image:
        # Single image
        inference_single_image(
            model, classes, args.image, device,
            args.input_size, args.save_dir, args.peak_threshold
        )
    else:
        # Directory
        inference_directory(
            model, classes, args.image_dir, device,
            args.input_size, args.save_dir, args.peak_threshold
        )


if __name__ == "__main__":
    main()
