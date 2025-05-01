# config.py
import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 256
VAL_EVERY_N_EPOCH   = 1
NUM_EPOCHS          = 40

# 기본 Optimizer (Adam)
OPTIMIZER_PARAMS = {
    'type': 'Adam',
    'lr': 0.001,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'weight_decay': 1e-4,
}

# 기본 Scheduler (CosineAnnealingLR)
SCHEDULER_PARAMS = {
    'type': 'CosineAnnealingLR',
    'T_max': NUM_EPOCHS,
    'eta_min': 1e-6,
}

# Dataset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network: 모델 목록
MODEL_LIST = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'efficientnet_b3',
    'efficientnet_b4',
    'efficientnet_b5',
    'efficientnet_b6',
    'efficientnet_b7',
]

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0]
PRECISION_STR       = '16-mixed'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
