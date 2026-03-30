import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'split_data')  # Updated to split dataset
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, 'train')
VAL_DIR = os.path.join(PROCESSED_DATA_DIR, 'val')
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, 'test')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'models', 'checkpoints')

# Model parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 64  # Increased for faster training
EPOCHS = 30
LEARNING_RATE = 0.001
TEST_SPLIT = 0.1

# Model file
MODEL_PATH = os.path.join(MODEL_DIR, 'skin_disease_model.h5')
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, 'class_indices.json')

# Disease information database
DISEASE_INFO_PATH = os.path.join(DATA_DIR, 'disease_info.json')

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)