"""
Central configuration for the ResNet-50 transfer learning pipeline.
Edit these values instead of hunting through the training script.
"""

import torch

# ---------------- Paths ----------------
DATA_DIR = "sample_data"          # expects DATA_DIR/<class_name>/*.jpg  (ImageFolder format)
OUTPUT_DIR = "outputs"
CHECKPOINT_PATH = f"{OUTPUT_DIR}/best_model.pt"

# ---------------- Data split ----------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42

# ---------------- Model ----------------
NUM_CLASSES = None                # inferred at runtime from the dataset
FREEZE_BACKBONE = False           # False = fine-tune all layers, True = feature-extraction only
IMAGE_SIZE = 224                  # standard ResNet-50 input size

# ---------------- Augmentation ----------------
RANDOM_CROP_PADDING = 4
COLOR_JITTER = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
NORM_MEAN = [0.485, 0.456, 0.406]  # ImageNet statistics (matches pretrained backbone)
NORM_STD = [0.229, 0.224, 0.225]

# ---------------- Training ----------------
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
OPTIMIZER = "adam"                 # "adam" or "sgd" (used by ablation.py for comparison)
LR_SCHEDULE = "cosine"             # "cosine" or "step" or "none" (used by ablation.py)
EARLY_STOP_PATIENCE = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2
