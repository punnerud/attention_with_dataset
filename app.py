#!/usr/bin/env python3
"""
Web-based Image Annotation Tool for Weakly-Supervised Object Detection
Allows you to go through images one by one and set class labels with counts.
Runs in your web browser - no tkinter required!
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory
import base64
import io
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

app = Flask(__name__)

# Paths
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
ANNOTATIONS_FILE = Path("data/annotations/annotations.json")
MODEL_PATH = Path("output/model.pth")

# Create directories
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Class definitions (customize these!)
CLASSES = ["blank", "bryter", "stikkontakt", "elsparkesykkel", "sluk", "kumlokk"]

# Global model variables
loaded_model = None
model_classes = None
sam_predictor = None
sam_available = False


def load_model():
    """Load trained model if available"""
    global loaded_model, model_classes

    if not MODEL_PATH.exists():
        return None

    try:
        from train import WeakCountModel

        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        num_classes = checkpoint['num_classes']
        model_classes = checkpoint['classes']

        model = WeakCountModel(num_classes).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        loaded_model = model
        print(f"✓ Model loaded: {model_classes}")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


def load_sam_model():
    """Load Segment Anything Model if available"""
    global sam_predictor, sam_available

    try:
        from segment_anything import sam_model_registry, SamPredictor

        # Try to find SAM checkpoint
        sam_checkpoint_paths = [
            "output/sam_vit_h_4b8939.pth",
            "output/sam_vit_l_0b3195.pth",
            "output/sam_vit_b_01ec64.pth",
        ]

        sam_checkpoint = None
        model_type = None

        for path in sam_checkpoint_paths:
            if Path(path).exists():
                sam_checkpoint = path
                if "vit_h" in path:
                    model_type = "vit_h"
                elif "vit_l" in path:
                    model_type = "vit_l"
                elif "vit_b" in path:
                    model_type = "vit_b"
                break

        if sam_checkpoint is None:
            print("ℹ SAM model not found. Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
            return None

        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device)

        sam_predictor = SamPredictor(sam)
        sam_available = True
        print(f"✓ SAM model loaded: {model_type}")
        return sam_predictor
    except ImportError:
        print("ℹ SAM not installed. Install with: pip install segment-anything")
        return None
    except Exception as e:
        print(f"Failed to load SAM: {e}")
        return None


def generate_attention_overlay(image_path, model):
    """Generate attention visualization for an image"""
    if model is None:
        return None

    try:
        device = next(model.parameters()).device

        # Load and preprocess image
        img_original = Image.open(image_path).convert('RGB')

        # Resize maintaining aspect ratio + pad to 448x448 (no cropping!)
        def resize_with_padding(img, target_size=448):
            """Resize image to fit in square, pad the rest"""
            w, h = img.size
            scale = target_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)

            # Resize
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Create padded image
            img_padded = Image.new('RGB', (target_size, target_size), (0, 0, 0))
            paste_x = (target_size - new_w) // 2
            paste_y = (target_size - new_h) // 2
            img_padded.paste(img_resized, (paste_x, paste_y))

            return img_padded, (paste_x, paste_y, new_w, new_h)

        img_processed, (pad_x, pad_y, img_w, img_h) = resize_with_padding(img_original, 448)

        # Convert to tensor for model
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img_tensor = to_tensor(img_processed).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            logits, den, counts = model(img_tensor)

        # Get predictions
        den = den[0].cpu().numpy()  # (C, h, w)
        counts = counts[0].cpu().numpy()  # (C,)

        # Create visualization in 3x2 grid (3 columns, 2 rows)
        num_classes = len(model_classes)
        ncols = 3
        nrows = (num_classes + ncols - 1) // ncols  # Ceiling division

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axes = axes.flatten() if num_classes > 1 else [axes]

        for i, class_name in enumerate(model_classes):
            dmap = den[i]
            count = counts[i]

            # Upsample density map to match processed image size (448x448)
            dmap_up = F.interpolate(
                torch.from_numpy(dmap).unsqueeze(0).unsqueeze(0),
                size=(448, 448),
                mode='bilinear',
                align_corners=False
            )[0, 0].numpy()

            # Normalize
            if dmap_up.max() > 0:
                dmap_up = (dmap_up - dmap_up.min()) / (dmap_up.max() - dmap_up.min())

            # Plot on processed image (not original)
            axes[i].imshow(img_processed)
            if dmap_up.max() > 0:
                axes[i].imshow(dmap_up, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            axes[i].set_title(f'{class_name}\nCount: {count:.1f}', fontsize=12)
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(num_classes, len(axes)):
            axes[i].axis('off')

        plt.tight_layout(pad=1.0)

        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return img_base64
    except Exception as e:
        print(f"Error generating attention: {e}")
        return None


def generate_sam_segmentation(image_path, model, image_name):
    """
    Generate SAM-based segmentation using attention maps and known classes
    This uses the attention maps as prompts to guide SAM segmentation
    """
    global sam_predictor

    if model is None or sam_predictor is None:
        return None

    try:
        # Load annotations to get known classes for this image
        annotations = load_annotations()
        if image_name not in annotations['images']:
            return None

        image_annotation = annotations['images'][image_name]
        active_classes = [cls for cls, count in image_annotation['counts'].items() if count > 0]

        if not active_classes:
            return None

        device = next(model.parameters()).device

        # Load and preprocess image
        img_original = Image.open(image_path).convert('RGB')
        img_np = np.array(img_original)

        # Resize maintaining aspect ratio + pad to 448x448
        def resize_with_padding(img, target_size=448):
            w, h = img.size
            scale = target_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img_padded = Image.new('RGB', (target_size, target_size), (0, 0, 0))
            paste_x = (target_size - new_w) // 2
            paste_y = (target_size - new_h) // 2
            img_padded.paste(img_resized, (paste_x, paste_y))
            return img_padded, (paste_x, paste_y, new_w, new_h)

        img_processed, (pad_x, pad_y, img_w, img_h) = resize_with_padding(img_original, 448)

        # Get attention maps from model
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img_tensor = to_tensor(img_processed).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, den, counts = model(img_tensor)

        den = den[0].cpu().numpy()  # (C, h, w)

        # Set SAM image
        sam_predictor.set_image(img_np)

        # Create visualization
        num_classes = len([cls for cls in model_classes if cls in active_classes])
        ncols = 3
        nrows = (num_classes + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axes = axes.flatten() if num_classes > 1 else [axes]

        ax_idx = 0
        for i, class_name in enumerate(model_classes):
            if class_name not in active_classes:
                continue

            dmap = den[i]

            # Upsample density map to original image size
            h_orig, w_orig = img_np.shape[:2]
            dmap_up = F.interpolate(
                torch.from_numpy(dmap).unsqueeze(0).unsqueeze(0),
                size=(h_orig, w_orig),
                mode='bilinear',
                align_corners=False
            )[0, 0].numpy()

            # Find top attention points as prompts for SAM
            threshold = np.percentile(dmap_up, 90)  # Top 10% attention
            point_coords = np.argwhere(dmap_up > threshold)

            if len(point_coords) == 0:
                axes[ax_idx].imshow(img_np)
                axes[ax_idx].set_title(f'{class_name}\n(No attention)', fontsize=12)
                axes[ax_idx].axis('off')
                ax_idx += 1
                continue

            # Sample up to 10 points (SAM works best with fewer points)
            if len(point_coords) > 10:
                indices = np.random.choice(len(point_coords), 10, replace=False)
                point_coords = point_coords[indices]

            # Convert to SAM format (x, y)
            sam_points = point_coords[:, [1, 0]]  # Swap to (x, y)
            point_labels = np.ones(len(sam_points))  # All positive prompts

            # Generate mask using SAM
            masks, scores, logits_sam = sam_predictor.predict(
                point_coords=sam_points,
                point_labels=point_labels,
                multimask_output=True
            )

            # Use the mask with highest score
            best_mask = masks[np.argmax(scores)]

            # Visualize
            axes[ax_idx].imshow(img_np)
            axes[ax_idx].imshow(best_mask, alpha=0.5, cmap='jet')

            # Show prompt points
            axes[ax_idx].scatter(sam_points[:, 0], sam_points[:, 1],
                               c='red', s=50, marker='*', edgecolors='white', linewidths=1)

            axes[ax_idx].set_title(f'{class_name}\n(SAM Segmentation)', fontsize=12)
            axes[ax_idx].axis('off')
            ax_idx += 1

        # Hide unused subplots
        for i in range(ax_idx, len(axes)):
            axes[i].axis('off')

        plt.tight_layout(pad=1.0)

        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return img_base64
    except Exception as e:
        print(f"Error generating SAM segmentation: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_annotations():
    """Load existing annotations or create new dict"""
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE, 'r') as f:
            return json.load(f)
    return {
        "classes": CLASSES,
        "images": {}
    }


def save_annotations(annotations):
    """Save annotations to JSON file"""
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(annotations, f, indent=2)


def get_image_list():
    """Get list of images in input directory"""
    return sorted([
        f.name for f in INPUT_DIR.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    ])


@app.route('/')
def index():
    """Serve the main annotation page"""
    return render_template('index.html')


@app.route('/api/config')
def get_config():
    """Get configuration (classes, image list)"""
    images = get_image_list()
    annotations = load_annotations()
    model_available = MODEL_PATH.exists()

    # Check SAM availability
    global sam_available

    return jsonify({
        'classes': CLASSES,
        'images': images,
        'total_images': len(images),
        'model_available': model_available,
        'sam_available': sam_available
    })


@app.route('/api/attention/<image_name>')
def get_attention(image_name):
    """Get attention visualization for an image"""
    global loaded_model

    if loaded_model is None:
        loaded_model = load_model()

    if loaded_model is None:
        return jsonify({'error': 'Model not available'}), 404

    image_path = INPUT_DIR / image_name
    if not image_path.exists():
        return jsonify({'error': 'Image not found'}), 404

    attention_img = generate_attention_overlay(image_path, loaded_model)

    if attention_img is None:
        return jsonify({'error': 'Failed to generate attention'}), 500

    return jsonify({'attention_image': attention_img})


@app.route('/api/sam-segmentation/<image_name>')
def get_sam_segmentation(image_name):
    """Get SAM-based segmentation using attention + known classes"""
    global loaded_model, sam_predictor

    if loaded_model is None:
        loaded_model = load_model()

    if sam_predictor is None:
        sam_predictor = load_sam_model()

    if loaded_model is None:
        return jsonify({'error': 'Model not available'}), 404

    if sam_predictor is None:
        return jsonify({'error': 'SAM not available'}), 404

    image_path = INPUT_DIR / image_name
    if not image_path.exists():
        return jsonify({'error': 'Image not found'}), 404

    sam_img = generate_sam_segmentation(image_path, loaded_model, image_name)

    if sam_img is None:
        return jsonify({'error': 'Failed to generate SAM segmentation'}), 500

    return jsonify({'sam_image': sam_img})


@app.route('/api/annotations')
def get_annotations():
    """Get all annotations"""
    annotations = load_annotations()
    return jsonify(annotations)


@app.route('/api/annotation/<image_name>')
def get_annotation(image_name):
    """Get annotation for specific image"""
    annotations = load_annotations()
    if image_name in annotations['images']:
        return jsonify(annotations['images'][image_name])
    return jsonify({
        'counts': {cls: 0 for cls in CLASSES},
        'classes': [],
        'primary_class': None
    })


@app.route('/api/annotation/<image_name>', methods=['POST'])
def save_annotation(image_name):
    """Save annotation for specific image"""
    annotations = load_annotations()
    data = request.json

    counts = data.get('counts', {})
    classes_present = [cls for cls, count in counts.items() if count > 0]

    # Determine primary class
    primary_class = None
    if classes_present:
        primary_class = max(classes_present, key=lambda c: counts[c])

    # Save annotation
    annotations['images'][image_name] = {
        'path': str(INPUT_DIR / image_name),
        'classes': classes_present,
        'primary_class': primary_class,
        'counts': counts
    }

    save_annotations(annotations)
    return jsonify({'status': 'success'})


@app.route('/api/save-all', methods=['POST'])
def save_all():
    """Save all annotations"""
    annotations = load_annotations()
    save_annotations(annotations)
    return jsonify({
        'status': 'success',
        'message': f'Saved {len(annotations["images"])} annotations'
    })


@app.route('/input/<path:filename>')
def serve_image(filename):
    """Serve images from input directory"""
    return send_from_directory(INPUT_DIR, filename)


def create_html_template():
    """Create the HTML template for the annotation interface"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Annotator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: #2c3e50;
            color: white;
            padding: 20px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header h1 {
            font-size: 24px;
        }

        .controls {
            background: #34495e;
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: white;
        }

        .nav-buttons button {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }

        .nav-buttons button:hover {
            background: #2980b9;
        }

        .nav-buttons button:disabled {
            background: #95a5a6;
            cursor: not-allowed;
        }

        .save-btn {
            background: #27ae60 !important;
            padding: 10px 30px !important;
        }

        .save-btn:hover {
            background: #229954 !important;
        }

        .main-content {
            display: flex;
            height: calc(100vh - 200px);
        }

        .image-panel {
            flex: 1;
            padding: 30px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #ecf0f1;
        }

        .image-container {
            max-width: 100%;
            max-height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .image-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 4px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        .annotation-panel {
            width: 400px;
            padding: 30px;
            border-left: 1px solid #ddd;
            overflow-y: auto;
        }

        .annotation-panel h2 {
            margin-bottom: 20px;
            color: #2c3e50;
        }

        .class-item {
            background: #f8f9fa;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 6px;
            border: 2px solid #e0e0e0;
        }

        .class-item.active {
            border-color: #3498db;
            background: #e3f2fd;
        }

        .class-header {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }

        .class-header input[type="checkbox"] {
            width: 20px;
            height: 20px;
            margin-right: 10px;
            cursor: pointer;
        }

        .class-header label {
            font-weight: 600;
            font-size: 16px;
            cursor: pointer;
            flex: 1;
        }

        .count-control {
            display: flex;
            align-items: center;
            margin-top: 10px;
        }

        .count-control label {
            margin-right: 10px;
            font-size: 14px;
            color: #666;
        }

        .count-control input[type="number"] {
            width: 80px;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }

        .progress-info {
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .filename {
            font-weight: 600;
            font-size: 16px;
        }

        .instructions {
            background: #fff3cd;
            padding: 15px;
            margin-top: 20px;
            border-radius: 4px;
            font-size: 14px;
            border-left: 4px solid #ffc107;
        }

        .instructions strong {
            display: block;
            margin-bottom: 8px;
        }

        .loading {
            text-align: center;
            padding: 50px;
            color: #666;
        }

        .no-images {
            text-align: center;
            padding: 50px;
            color: #e74c3c;
        }

        .stats {
            background: #e8f5e9;
            padding: 10px 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            font-size: 14px;
        }

        .attention-toggle {
            background: #fff;
            padding: 10px 15px;
            margin-bottom: 15px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            gap: 10px;
            border: 2px solid #ddd;
        }

        .attention-toggle input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }

        .attention-toggle label {
            cursor: pointer;
            font-weight: 600;
        }

        .attention-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: none;
            align-items: center;
            justify-content: center;
            background: rgba(255, 255, 255, 0.95);
        }

        .attention-overlay.visible {
            display: flex;
        }

        .attention-overlay img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }

        .loading-spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🖼️ Image Annotator</h1>
            <div class="stats" id="stats">Loading...</div>
        </div>

        <div class="controls">
            <div class="progress-info">
                <span id="progress">Image 0 / 0</span>
                <span class="filename" id="filename">No image</span>
            </div>
            <div class="nav-buttons">
                <button id="prevBtn" onclick="previousImage()">← Previous</button>
                <button id="nextBtn" onclick="nextImage()">Next →</button>
                <button class="save-btn" onclick="saveAll()">💾 Save All</button>
            </div>
        </div>

        <div class="main-content">
            <div class="image-panel">
                <div class="image-container" id="imageContainer" style="position: relative;">
                    <div class="loading">Loading...</div>
                    <div class="attention-overlay" id="attentionOverlay">
                        <div class="loading-spinner"></div>
                    </div>
                </div>
            </div>

            <div class="annotation-panel">
                <h2>Annotations</h2>

                <!-- Attention toggle -->
                <div class="attention-toggle" id="attentionToggleContainer" style="display: none;">
                    <input type="checkbox" id="showAttention" onchange="toggleAttention()">
                    <label for="showAttention">🔥 Show Attention Map</label>
                </div>

                <!-- SAM Segmentation toggle -->
                <div class="attention-toggle" id="samToggleContainer" style="display: none;">
                    <input type="checkbox" id="showSAM" onchange="toggleSAM()">
                    <label for="showSAM">✂️ Show SAM Segmentation</label>
                </div>

                <div id="classControls"></div>

                <div class="instructions">
                    <strong>Instructions:</strong>
                    • Check classes present in the image<br>
                    • Set count for each class<br>
                    • Use arrow keys or buttons to navigate<br>
                    • Changes are saved automatically<br>
                    • Click "Save All" when done
                </div>
            </div>
        </div>
    </div>

    <script>
        let config = null;
        let images = [];
        let currentIndex = 0;
        let annotations = {};

        // Initialize
        async function init() {
            try {
                const configRes = await fetch('/api/config');
                config = await configRes.json();
                images = config.images;

                const annotationsRes = await fetch('/api/annotations');
                annotations = await annotationsRes.json();

                if (images.length === 0) {
                    document.getElementById('imageContainer').innerHTML =
                        '<div class="no-images">No images found in input/ folder.<br>Please add images and refresh.</div>';
                    return;
                }

                createClassControls();

                // Show attention toggle if model is available
                if (config.model_available) {
                    document.getElementById('attentionToggleContainer').style.display = 'flex';
                }

                // Show SAM toggle if SAM is available
                if (config.sam_available) {
                    document.getElementById('samToggleContainer').style.display = 'flex';
                }

                // Find first unannotated image or start at 0
                let startIndex = 0;
                for (let i = 0; i < images.length; i++) {
                    if (!annotations.images[images[i]]) {
                        startIndex = i;
                        break;
                    }
                }

                loadImage(startIndex);
                updateStats();
            } catch (error) {
                console.error('Failed to initialize:', error);
                alert('Failed to load configuration. Make sure the server is running.');
            }
        }

        function createClassControls() {
            const container = document.getElementById('classControls');
            container.innerHTML = '';

            config.classes.forEach(className => {
                const div = document.createElement('div');
                div.className = 'class-item';
                div.id = `class-${className}`;

                div.innerHTML = `
                    <div class="class-header">
                        <input type="checkbox"
                               id="cb-${className}"
                               onchange="toggleClass('${className}')">
                        <label for="cb-${className}">${className.replace(/_/g, ' ').toUpperCase()}</label>
                    </div>
                    <div class="count-control">
                        <label>Count:</label>
                        <input type="number"
                               id="count-${className}"
                               min="0"
                               max="100"
                               value="0"
                               onchange="updateCount('${className}')">
                    </div>
                `;

                container.appendChild(div);
            });
        }

        async function loadImage(index) {
            if (index < 0 || index >= images.length) return;

            currentIndex = index;
            const imageName = images[index];

            // Update UI
            document.getElementById('progress').textContent = `Image ${index + 1} / ${images.length}`;
            document.getElementById('filename').textContent = imageName;

            // Load image
            const img = document.createElement('img');
            img.src = `/input/${imageName}`;
            img.alt = imageName;
            const container = document.getElementById('imageContainer');
            container.innerHTML = '';
            container.appendChild(img);

            // Re-add attention overlay
            const overlay = document.createElement('div');
            overlay.className = 'attention-overlay';
            overlay.id = 'attentionOverlay';
            container.appendChild(overlay);

            // Reset attention and SAM checkboxes
            const attentionCheckbox = document.getElementById('showAttention');
            const samCheckbox = document.getElementById('showSAM');
            if (attentionCheckbox) {
                attentionCheckbox.checked = false;
            }
            if (samCheckbox) {
                samCheckbox.checked = false;
            }

            // Load annotation
            try {
                const res = await fetch(`/api/annotation/${imageName}`);
                const annotation = await res.json();

                config.classes.forEach(className => {
                    const count = annotation.counts[className] || 0;
                    const checkbox = document.getElementById(`cb-${className}`);
                    const countInput = document.getElementById(`count-${className}`);
                    const classItem = document.getElementById(`class-${className}`);

                    checkbox.checked = count > 0;
                    countInput.value = count;

                    if (count > 0) {
                        classItem.classList.add('active');
                    } else {
                        classItem.classList.remove('active');
                    }
                });
            } catch (error) {
                console.error('Failed to load annotation:', error);
            }

            // Update navigation buttons
            document.getElementById('prevBtn').disabled = index === 0;
            document.getElementById('nextBtn').disabled = index === images.length - 1;
        }

        function toggleClass(className) {
            const checkbox = document.getElementById(`cb-${className}`);
            const countInput = document.getElementById(`count-${className}`);
            const classItem = document.getElementById(`class-${className}`);

            if (checkbox.checked) {
                if (countInput.value == 0) {
                    countInput.value = 1;
                }
                classItem.classList.add('active');
            } else {
                countInput.value = 0;
                classItem.classList.remove('active');
            }

            saveCurrentAnnotation();
        }

        function updateCount(className) {
            const countInput = document.getElementById(`count-${className}`);
            const checkbox = document.getElementById(`cb-${className}`);
            const classItem = document.getElementById(`class-${className}`);
            const count = parseInt(countInput.value) || 0;

            checkbox.checked = count > 0;

            if (count > 0) {
                classItem.classList.add('active');
            } else {
                classItem.classList.remove('active');
            }

            saveCurrentAnnotation();
        }

        async function saveCurrentAnnotation() {
            if (!images[currentIndex]) return;

            const imageName = images[currentIndex];
            const counts = {};

            config.classes.forEach(className => {
                const countInput = document.getElementById(`count-${className}`);
                counts[className] = parseInt(countInput.value) || 0;
            });

            try {
                await fetch(`/api/annotation/${imageName}`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({counts})
                });
                updateStats();
            } catch (error) {
                console.error('Failed to save annotation:', error);
            }
        }

        async function saveAll() {
            try {
                const res = await fetch('/api/save-all', {method: 'POST'});
                const data = await res.json();
                alert(data.message);
            } catch (error) {
                console.error('Failed to save:', error);
                alert('Failed to save annotations');
            }
        }

        async function updateStats() {
            try {
                const res = await fetch('/api/annotations');
                const data = await res.json();
                const annotated = Object.keys(data.images).length;
                const total = images.length;
                document.getElementById('stats').textContent =
                    `📊 Annotated: ${annotated} / ${total} images`;
            } catch (error) {
                console.error('Failed to update stats:', error);
            }
        }

        function previousImage() {
            if (currentIndex > 0) {
                loadImage(currentIndex - 1);
            }
        }

        function nextImage() {
            if (currentIndex < images.length - 1) {
                loadImage(currentIndex + 1);
            }
        }

        async function toggleAttention() {
            const checkbox = document.getElementById('showAttention');
            const samCheckbox = document.getElementById('showSAM');
            const overlay = document.getElementById('attentionOverlay');

            // Uncheck SAM if attention is being turned on
            if (checkbox.checked && samCheckbox) {
                samCheckbox.checked = false;
            }

            if (checkbox.checked) {
                // Show attention
                overlay.classList.add('visible');
                overlay.innerHTML = '<div class="loading-spinner"></div>';

                try {
                    const imageName = images[currentIndex];
                    const res = await fetch(`/api/attention/${imageName}`);
                    const data = await res.json();

                    if (data.attention_image) {
                        overlay.innerHTML = `<img src="data:image/png;base64,${data.attention_image}" alt="Attention Map">`;
                    } else {
                        overlay.innerHTML = '<div style="color: red;">Failed to load attention map</div>';
                    }
                } catch (error) {
                    console.error('Failed to load attention:', error);
                    overlay.innerHTML = '<div style="color: red;">Error loading attention</div>';
                }
            } else {
                // Hide attention
                overlay.classList.remove('visible');
            }
        }

        async function toggleSAM() {
            const checkbox = document.getElementById('showSAM');
            const attentionCheckbox = document.getElementById('showAttention');
            const overlay = document.getElementById('attentionOverlay');

            // Uncheck attention if SAM is being turned on
            if (checkbox.checked && attentionCheckbox) {
                attentionCheckbox.checked = false;
            }

            if (checkbox.checked) {
                // Show SAM segmentation
                overlay.classList.add('visible');
                overlay.innerHTML = '<div class="loading-spinner"></div>';

                try {
                    const imageName = images[currentIndex];
                    const res = await fetch(`/api/sam-segmentation/${imageName}`);
                    const data = await res.json();

                    if (data.sam_image) {
                        overlay.innerHTML = `<img src="data:image/png;base64,${data.sam_image}" alt="SAM Segmentation">`;
                    } else if (data.error) {
                        overlay.innerHTML = `<div style="color: red;">${data.error}</div>`;
                    } else {
                        overlay.innerHTML = '<div style="color: red;">Failed to load SAM segmentation</div>';
                    }
                } catch (error) {
                    console.error('Failed to load SAM:', error);
                    overlay.innerHTML = '<div style="color: red;">Error loading SAM segmentation</div>';
                }
            } else {
                // Hide SAM
                overlay.classList.remove('visible');
            }
        }

        // Keyboard shortcuts
        let activeClass = null;

        document.addEventListener('keydown', (e) => {
            // Ignore if typing in a number input
            if (e.target.type === 'number') {
                return;
            }

            const key = e.key.toLowerCase();

            // Navigation shortcuts
            if (e.key === 'ArrowLeft' || key === 'p') {
                e.preventDefault();
                previousImage();
            } else if (e.key === 'ArrowRight' || key === 'n') {
                e.preventDefault();
                nextImage();
            } else if (e.ctrlKey && key === 's') {
                e.preventDefault();
                saveAll();
            }
            // Class shortcuts: Space/0, B, K, E, S, M (for kuMlokk)
            else if (key === ' ' || key === '0') {
                e.preventDefault();
                focusClass('blank');
            } else if (key === 'b') {
                e.preventDefault();
                focusClass('bryter');
            } else if (key === 'k') {
                e.preventDefault();
                focusClass('stikkontakt');
            } else if (key === 'e') {
                e.preventDefault();
                focusClass('elsykkel');
            } else if (key === 's') {
                e.preventDefault();
                focusClass('sluk');
            } else if (key === 'm') {
                e.preventDefault();
                focusClass('kumlokk');
            }
            // Number keys to set count for active class
            else if (activeClass && key >= '1' && key <= '9') {
                e.preventDefault();
                const countInput = document.getElementById(`count-${activeClass}`);
                countInput.value = key;
                updateCount(activeClass);
                // Clear active class after setting number
                const classItem = document.getElementById(`class-${activeClass}`);
                if (classItem) {
                    classItem.style.outline = 'none';
                }
                activeClass = null;
            }
            // Backspace to clear count for active class
            else if (activeClass && e.key === 'Backspace') {
                e.preventDefault();
                const countInput = document.getElementById(`count-${activeClass}`);
                const currentValue = countInput.value || '0';
                if (currentValue.length > 1) {
                    countInput.value = currentValue.slice(0, -1);
                } else {
                    countInput.value = '0';
                }
                updateCount(activeClass);
            }
        });

        function focusClass(className) {
            // Remove previous active state
            if (activeClass) {
                const prevItem = document.getElementById(`class-${activeClass}`);
                if (prevItem) {
                    prevItem.style.outline = 'none';
                }
            }

            // Set new active class
            activeClass = className;
            const classItem = document.getElementById(`class-${className}`);
            const countInput = document.getElementById(`count-${className}`);
            const checkbox = document.getElementById(`cb-${className}`);

            // Visual feedback
            classItem.style.outline = '3px solid #f39c12';
            classItem.style.outlineOffset = '2px';

            // Toggle checkbox
            checkbox.checked = !checkbox.checked;

            // If checked and count is 0, set to 1
            if (checkbox.checked && countInput.value == '0') {
                countInput.value = '1';
            } else if (!checkbox.checked) {
                countInput.value = '0';
            }

            // Update class item appearance
            if (checkbox.checked) {
                classItem.classList.add('active');
            } else {
                classItem.classList.remove('active');
            }

            saveCurrentAnnotation();

            // Remove outline after 2 seconds
            setTimeout(() => {
                if (activeClass === className) {
                    classItem.style.outline = 'none';
                    activeClass = null;
                }
            }, 2000);
        }

        // Start the app
        init();
    </script>
</body>
</html>
"""

    with open(templates_dir / "index.html", 'w') as f:
        f.write(html_content)


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Image Annotation Tool")
    print("=" * 60)
    print(f"📁 Input directory: {INPUT_DIR.absolute()}")
    print(f"💾 Annotations file: {ANNOTATIONS_FILE.absolute()}")
    print(f"📊 Classes: {', '.join(CLASSES)}")
    print("=" * 60)

    # Create HTML template
    create_html_template()

    # Get image count
    images = get_image_list()
    print(f"🖼️  Found {len(images)} images")

    if len(images) == 0:
        print("\n⚠️  WARNING: No images found in input/ folder!")
        print("   Please add images (.jpg, .jpeg, .png, .bmp) and restart.")

    print("\n🌐 Starting web server...")
    print("   Open your browser and go to: http://localhost:8100")
    print("\n   Press Ctrl+C to stop the server")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=8100)
