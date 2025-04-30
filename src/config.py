import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 512
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 40

# original SGD
# OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.005*5, 'momentum': 0.9}

# Adam
OPTIMIZER_PARAMS = {
    'type': 'Adam',
    'lr': 0.001*2,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'weight_decay': 2e-4  # overfit 방지용 / 기본값 0, 보통 1e-4정도 사용 
}

# original Scheduler
# SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2}

# CosineAnnealingLR
SCHEDULER_PARAMS = {
    'type': 'CosineAnnealingLR',
    'T_max': 200  # 전체 epoch 수
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

# Network
MODEL_NAME          = 'resnet34'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0]
PRECISION_STR       = '16-mixed' # 32-fixed -> 16-mixed

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
