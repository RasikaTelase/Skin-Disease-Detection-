
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import config

def create_data_generators(train_dir, img_height, img_width, batch_size):
    """
    Create data generators for training and validation
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=config.VALIDATION_SPLIT
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=config.VALIDATION_SPLIT
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def prepare_dataset_structure(source_dir, dest_dir):
    """
    Organize dataset into proper structure for training
    The Dermnet dataset should already be organized by disease folders
    """
    print("Preparing dataset structure...")
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} not found!")
        print("Please download the Dermnet dataset from Kaggle first.")
        print("\nSteps to download:")
        print("1. Install kaggle: pip install kaggle")
        print("2. Set up Kaggle API credentials (kaggle.json)")
        print("3. Run: kaggle datasets download -d shubhamgoel27/dermnet")
        print("4. Extract to data/raw/ folder")
        return False
    
    # Check if dataset is already organized
    subdirs = [d for d in os.listdir(source_dir) 
               if os.path.isdir(os.path.join(source_dir, d))]
    
    if len(subdirs) > 0:
        print(f"Found {len(subdirs)} disease categories")
        
        # Copy organized dataset to processed folder
        if not os.path.exists(dest_dir):
            print("Copying dataset to processed folder...")
            shutil.copytree(source_dir, dest_dir)
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        print("-" * 50)
        total_images = 0
        for disease_folder in sorted(subdirs):
            disease_path = os.path.join(dest_dir, disease_folder)
            if os.path.isdir(disease_path):
                num_images = len([f for f in os.listdir(disease_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"{disease_folder}: {num_images} images")
                total_images += num_images
        
        print("-" * 50)
        print(f"Total images: {total_images}")
        print(f"Total classes: {len(subdirs)}")
        
        return True
    else:
        print("Dataset structure not recognized. Please check the dataset format.")
        return False

def load_and_preprocess_image(image_path, img_height, img_width):
    """
    Load and preprocess a single image for prediction
    """
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

def get_class_weights(train_generator):
    """
    Calculate class weights for imbalanced dataset
    """
    counter = train_generator.classes
    class_counts = np.bincount(counter)
    total = sum(class_counts)
    
    class_weight = {}
    for i, count in enumerate(class_counts):
        class_weight[i] = total / (len(class_counts) * count)
    
    return class_weight