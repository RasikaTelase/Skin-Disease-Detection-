
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def load_disease_info(disease_info_path):
    """Load disease information from JSON file"""
    with open(disease_info_path, 'r') as f:
        return json.load(f)

def save_class_indices(class_indices, filepath):
    """Save class indices to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(class_indices, f, indent=4)

def load_class_indices(filepath):
    """Load class indices from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def print_classification_report(y_true, y_pred, class_names):
    """Print detailed classification report"""
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print("=" * 80)
    print(report)

def create_directory_structure(base_dir):
    """Create necessary directories for the project"""
    dirs = [
        'data/raw',
        'data/processed',
        'models/saved_models',
        'models/checkpoints',
        'notebooks',
        'src',
        'web_app/static/css',
        'web_app/static/js',
        'web_app/static/uploads',
        'web_app/templates'
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
    
    print("Directory structure created successfully!")

def get_disease_info(disease_name, disease_info_dict):
    """Get information about a specific disease"""
    return disease_info_dict.get(disease_name, {
        "description": "Information not available",
        "symptoms": [],
        "causes": [],
        "treatment": [],
        "prevention": []
    })