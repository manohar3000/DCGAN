import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data parameters
DATASET_PATH = "D:/img_align_celeba"  # Update this path as needed
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 128
NUM_WORKERS = 0  # Increase if you want faster data loading

# Model parameters
Z_DIM = 100  # Dimension of latent space
LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)

# Training parameters
NUM_EPOCHS = 5
SAVE_IMAGE_INTERVAL = 500  # iterations
FIXED_NOISE_SIZE = 64

# Output directories
SAVED_IMAGES_DIR = "saved_images"
