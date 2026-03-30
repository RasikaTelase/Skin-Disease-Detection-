
import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)

# Paths
BASE_DIR = Path(__file__).parent.parent
SOURCE_DIR = BASE_DIR / 'data' / 'IMG_CLASSES'
OUTPUT_DIR = BASE_DIR / 'data' / 'split_data'

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

print("=" * 80)
print("DATA SPLITTING SCRIPT")
print("=" * 80)
print(f"\nSource Directory: {SOURCE_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"\nSplit Ratios:")
print(f"  Train: {TRAIN_RATIO*100:.0f}%")
print(f"  Validation: {VAL_RATIO*100:.0f}%")
print(f"  Test: {TEST_RATIO*100:.0f}%")
print("=" * 80)

# Create output directories
for split in ['train', 'val', 'test']:
    split_dir = OUTPUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {split_dir}")

# Get all class folders
class_folders = [f for f in SOURCE_DIR.iterdir() if f.is_dir()]
print(f"\n✓ Found {len(class_folders)} classes")

# Process each class
total_train = 0
total_val = 0
total_test = 0

for class_folder in class_folders:
    class_name = class_folder.name
    print(f"\n📁 Processing: {class_name}")
    
    # Create class subdirectories in train/val/test
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Get all images in this class
    images = list(class_folder.glob('*'))
    images = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    # Shuffle images
    random.shuffle(images)
    
    # Calculate split indices
    total_images = len(images)
    train_end = int(total_images * TRAIN_RATIO)
    val_end = train_end + int(total_images * VAL_RATIO)
    
    # Split images
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    print(f"   Total: {total_images} images")
    print(f"   Train: {len(train_images)} | Val: {len(val_images)} | Test: {len(test_images)}")
    
    # Copy images to respective folders
    for img in tqdm(train_images, desc="   Train", ncols=80):
        shutil.copy2(img, OUTPUT_DIR / 'train' / class_name / img.name)
    
    for img in tqdm(val_images, desc="   Val  ", ncols=80):
        shutil.copy2(img, OUTPUT_DIR / 'val' / class_name / img.name)
    
    for img in tqdm(test_images, desc="   Test ", ncols=80):
        shutil.copy2(img, OUTPUT_DIR / 'test' / class_name / img.name)
    
    total_train += len(train_images)
    total_val += len(val_images)
    total_test += len(test_images)

print("\n" + "=" * 80)
print("DATA SPLITTING COMPLETED!")
print("=" * 80)
print(f"\n📊 FINAL STATISTICS:")
print(f"   Training:   {total_train:,} images ({total_train/(total_train+total_val+total_test)*100:.1f}%)")
print(f"   Validation: {total_val:,} images ({total_val/(total_train+total_val+total_test)*100:.1f}%)")
print(f"   Test:       {total_test:,} images ({total_test/(total_train+total_val+total_test)*100:.1f}%)")
print(f"   Total:      {total_train+total_val+total_test:,} images")
print(f"\n📁 Output Location: {OUTPUT_DIR}")
print("=" * 80)