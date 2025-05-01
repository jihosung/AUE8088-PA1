# config.py (수정)
import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 512
VAL_EVERY_N_EPOCH   = 1
NUM_EPOCHS          = 40

# 기본 Optimizer (SGD)
OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9}

# 기본 Scheduler (MultiStepLR)
SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2}

# <실험용 하이퍼파라미터 리스트>
# 1) learning rate 비교 (기본 0.005 외에 0.01, 0.05)
LR_LIST = [0.005, 0.01, 0.05]

# 2) momentum 비교 (기본 0.9 외에 0.1, 1.5)
MOMENTUM_LIST = [0.9, 0.1, 1.5]

# 3) scheduler milestones 비교 ([30,35], [20,30], gamma는 그대로 0.2)
SCHEDULER_LIST = [
    {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2},
    {'type': 'MultiStepLR', 'milestones': [20, 30], 'gamma': 0.2},
]

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
MODEL_NAME          = 'resnet18'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0]
PRECISION_STR       = '16-mixed'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
