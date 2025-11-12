#!/usr/bin/env python3
"""
Rename 'elsykkel' to 'elsparkesykkel' in all annotation files
"""

import json
from pathlib import Path

def rename_class_in_annotations(filepath, old_name, new_name):
    """Rename a class in an annotations JSON file"""
    print(f"Processing {filepath}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Update classes list
    if 'classes' in data:
        data['classes'] = [new_name if c == old_name else c for c in data['classes']]

    # Update each image's annotations
    if 'images' in data:
        for img_name, img_data in data['images'].items():
            # Update classes list in image
            if 'classes' in img_data:
                img_data['classes'] = [new_name if c == old_name else c for c in img_data['classes']]

            # Update primary_class
            if img_data.get('primary_class') == old_name:
                img_data['primary_class'] = new_name

            # Update counts dictionary
            if 'counts' in img_data and old_name in img_data['counts']:
                img_data['counts'][new_name] = img_data['counts'].pop(old_name)

    # Save back
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ Updated {filepath}")

# Find and process all annotation files
annotation_files = [
    Path("data/annotations/annotations.json"),
    Path("data/annotations/annotations_backup.json"),
    Path("data/annotations/annotations_merged.json"),
]

for filepath in annotation_files:
    if filepath.exists():
        rename_class_in_annotations(filepath, "elsykkel", "elsparkesykkel")
    else:
        print(f"⚠ Skipping {filepath} (not found)")

print("\n✓ All annotation files updated!")
print("Note: Any trained models will still use the old class name internally.")
print("You may need to retrain the model for consistency.")
