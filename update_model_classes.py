#!/usr/bin/env python3
"""
Update class names in trained model checkpoint
"""

import torch
from pathlib import Path

model_path = Path('output/model.pth')

if not model_path.exists():
    print(f"❌ Model not found at {model_path}")
    exit(1)

# Create backup
backup_path = Path('output/model_backup.pth')
import shutil
shutil.copy(model_path, backup_path)
print(f"✓ Backup saved to {backup_path}")

# Load checkpoint
checkpoint = torch.load(model_path, map_location='cpu')

# Update classes
old_classes = checkpoint.get('classes', [])
print(f"\nOld classes: {old_classes}")

new_classes = ['elsparkesykkel' if c == 'elsykkel' else c for c in old_classes]
checkpoint['classes'] = new_classes

print(f"New classes: {new_classes}")

# Save updated checkpoint
torch.save(checkpoint, model_path)
print(f"\n✓ Model updated at {model_path}")
print("\nThe model weights remain unchanged - only the class labels were updated.")
print("Attention maps will now show 'elsparkesykkel' instead of 'elsykkel'!")
