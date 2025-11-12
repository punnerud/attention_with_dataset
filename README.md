# Weakly-Supervised Object Detection with Attention Maps

A web-based annotation tool and training pipeline for weakly-supervised object detection using image-level labels and count annotations.

## 📋 Overview

This project allows you to:
- Annotate images with class labels and object counts (no bounding boxes needed!)
- Train a deep learning model that learns to localize objects from weak supervision
- Visualize attention maps showing where the model "looks" for each class
- Export annotated datasets for further use

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip3 install flask torch torchvision pillow numpy matplotlib
```

Or if you have a `requirements.txt`:

```bash
pip3 install -r requirements.txt
```

### 2. Add Your Images

Place your images in the `input/` folder:

```bash
cp /path/to/your/images/*.jpg input/
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

### 3. Run the Annotation Tool

Start the web-based annotation interface:

```bash
python3 app.py
```

Then open your browser and go to:
```
http://localhost:8100
```

### 4. Annotate Your Images

The annotation interface allows you to:

#### Basic Workflow
1. **View images**: Navigate through images using arrow keys (←/→) or Next/Previous buttons
2. **Select classes**: Check the boxes for classes present in the image
3. **Set counts**: Enter the number of objects for each class (e.g., 3 outlets, 2 drains)
4. **Auto-save**: Annotations are saved automatically as you work
5. **Manual save**: Click "Save All" to ensure everything is persisted

#### Keyboard Shortcuts
- `←` or `P`: Previous image
- `→` or `N`: Next image
- `Ctrl+S`: Save all annotations
- `Space` or `0`: Toggle "blank" class
- `B`: Toggle "bryter" (switch)
- `K`: Toggle "stikkontakt" (outlet)
- `E`: Toggle "elsparkesykkel" (e-scooter)
- `S`: Toggle "sluk" (drain)
- `M`: Toggle "kumlokk" (manhole cover)
- `1-9`: Set count for active class

#### Tips
- If an image contains no objects of interest, just leave all counts at 0 or check "blank"
- You can annotate partially - the tool tracks which images have been annotated
- The progress indicator shows "Annotated: X / Y images" in the top right

### 5. Train the Model

Once you have annotated enough images (recommended: 50+ images minimum), train the model:

```bash
python3 train.py
```

**Training options:**
```bash
# Basic training with default settings
python3 train.py

# Specify number of epochs
python3 train.py --epochs 50

# Adjust batch size (reduce if out of memory)
python3 train.py --batch-size 4

# Change learning rate
python3 train.py --lr 0.0001
```

**What happens during training:**
- The model learns from image-level labels and counts
- Creates a weakly-supervised detector that localizes objects
- Saves the best model to `output/model.pth`
- Generates training plots in `output/training_plots.png`

**Training progress:**
```
Epoch [10/30], Loss: 2.3456, Count MAE: 1.23
Epoch [20/30], Loss: 1.2345, Count MAE: 0.89
...
✓ Training complete! Model saved to output/model.pth
```

### 6. View Attention Maps

After training, restart the annotation tool:

```bash
python3 app.py
```

Now you'll see a **"Show Attention Map"** toggle in the annotation panel. Check it to visualize where the model is looking for each class!

**Attention maps show:**
- Hot spots (red/yellow) where the model detects objects
- Cool areas (blue) with low confidence
- Separate heatmap for each class
- Predicted counts for each class

## 📁 Project Structure

```
attention_with_dataset/
├── app.py                      # Web annotation tool
├── train.py                    # Training script
├── inference.py                # Run inference on new images
├── dynamic_dataset.py          # Dataset loader
├── input/                      # Your images go here
│   ├── IMG_001.jpg
│   ├── IMG_002.jpg
│   └── ...
├── data/
│   └── annotations/
│       └── annotations.json    # Saved annotations
├── output/                     # Training outputs
│   ├── model.pth              # Trained model
│   └── training_plots.png     # Loss/accuracy plots
└── templates/
    └── index.html             # Auto-generated UI
```

## 🔄 Workflow Summary

```
1. Add images → input/
2. Run app.py → Annotate images
3. Run train.py → Train model
4. Run app.py again → View attention maps
5. Add more images → Repeat!
```

## 📸 Adding New Images

### Option 1: Add to Existing Dataset

Simply copy new images to the `input/` folder:

```bash
cp /path/to/new/images/*.jpg input/
```

Then restart `app.py` and annotate the new images.

### Option 2: Batch Import

If you have many images:

```bash
# Copy all images at once
cp -r /path/to/image/folder/*.jpg input/

# Or use a loop for different formats
for img in /path/to/images/*.{jpg,png}; do
    cp "$img" input/
done
```

### Option 3: Reduce Image Sizes (Recommended)

If your images are large, compress them first to save disk space:

```bash
# This was done using the reduce_images.py script
# which converts images to JPEG and reduces file size by ~75-85%
```

**Note:** If you convert images from PNG to JPG after annotation, make sure to update annotation references (already handled in this project).

## 🎯 Annotation Best Practices

### How Many Images?
- **Minimum**: 50 images per class
- **Good**: 100-200 images per class
- **Excellent**: 500+ images per class

### Quality Tips
1. **Be consistent**: Count all visible objects, even partial ones
2. **Verify counts**: Double-check your counts before moving to next image
3. **Handle occlusion**: Count partially visible objects if >50% visible
4. **Background images**: Include images with no objects (all counts = 0)
5. **Variety**: Include different angles, lighting, distances

### Class Balance
Try to have roughly equal numbers of images for each class. If one class has 200 images and another has 20, the model may perform poorly on the underrepresented class.

## 🔧 Customizing Classes

To change the classes, edit `app.py`:

```python
# Line 37 - Change these to your classes
CLASSES = ["blank", "bryter", "stikkontakt", "elsparkesykkel", "sluk", "kumlokk"]
```

Change to:
```python
CLASSES = ["person", "car", "bicycle", "dog", "cat"]
```

Also update keyboard shortcuts (lines 918-936) if desired.

## 🧪 Running Inference

To run the trained model on new images:

```bash
python3 inference.py --image path/to/image.jpg
```

This will:
- Load the trained model
- Generate attention maps
- Display predicted counts for each class
- Save visualization to `output/inference_result.png`

**Batch inference:**
```bash
# Process all images in a folder
python3 inference.py --folder path/to/images/
```

## 📊 Understanding the Model

### Architecture
- **Backbone**: ResNet-based feature extractor
- **Attention**: Learns spatial attention maps for each class
- **Counting**: Integrates attention to predict object counts
- **Supervision**: Trained only on image-level counts (no bounding boxes!)

### Loss Function
The model optimizes:
- **Count loss**: MSE between predicted and true counts
- **Attention regularization**: Encourages focused, localized attention

### Output
For each image and class:
- **Density map**: Spatial probability distribution of objects
- **Count**: Predicted number of objects (sum of density map)
- **Attention overlay**: Visualization of where model is looking

## 🐛 Troubleshooting

### Issue: "No images found"
**Solution**: Add images to `input/` folder and refresh the browser

### Issue: Annotations not showing after restart
**Solution**: Annotations are saved to `data/annotations/annotations.json`. Check if the file exists and contains your data.

### Issue: Out of memory during training
**Solution**: Reduce batch size:
```bash
python3 train.py --batch-size 2
```

### Issue: Model not loading in app.py
**Solution**: Make sure `output/model.pth` exists. Train the model first with `python3 train.py`

### Issue: Poor attention maps
**Solution**:
- Annotate more images (100+ per class)
- Train for more epochs: `python3 train.py --epochs 50`
- Check annotation quality - are counts accurate?

### Issue: Images changed from PNG to JPG
**Solution**: Already handled! Annotation files automatically updated to reference `.jpg` instead of `.png`

## 📚 Advanced Usage

### Data Augmentation

The training pipeline includes augmentation:
- Random horizontal flips
- Color jittering
- Random crops
- Normalization

Edit `dynamic_dataset.py` to customize augmentation.

### Composite Images

Create 2x2 grids of images for efficient annotation:

```bash
python3 augment_composite.py
```

This combines 4 images into larger grids, useful for small objects.

### Model Architecture

To modify the model architecture, edit `train.py`:
- Change backbone network
- Adjust attention mechanism
- Modify loss functions
- Add regularization

## 🎓 Citation

If you use this tool for research, please cite:

```
Weakly-Supervised Object Detection with Count Annotations
Morten Punnerud-Engelstad
2024
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2024 Morten Punnerud-Engelstad

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions or issues, please [create an issue](link-to-issues) or contact [your-email].

---

**Happy annotating! 🎉**
