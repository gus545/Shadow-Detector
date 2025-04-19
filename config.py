import torch
import os


# Path to the directory containing the training images.
TRAIN_IMAGE_DATASET_PATH = 'ISTD_Dataset/train/train_A'

# Path to the directory containing the test images.
TEST_IMAGE_DATASET_PATH = 'ISTD_Dataset/test/test_A'

# Path to the directory containing the training masks.
TRAIN_MASK_DATASET_PATH = 'ISTD_Dataset/train/train_B'

# Path to the directory containing the test masks.
TEST_MASK_DATASET_PATH = 'ISTD_Dataset/test/test_B'

TRAIN_REMOVED_DATASET_PATH = 'ISTD_Dataset/train/train_C'
TEST_REMOVED_DATASET_PATH = 'ISTD_Dataset/test/test_C'


# define training hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 15
INIT_LR = 0.0002
NUM_WORKERS = 4

# define transform dimensions
IMAGE_SIZE = 128


# define base output directory
BASE_OUTPUT_DIR = 'output'

# define model output directory
MODEL_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'models')

# define image output directory
IMAGE_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'images')

# define checkpoint output directory
CHECKPOINT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'checkpoints')

# define log output directory
LOG_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'logs')

# define device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


