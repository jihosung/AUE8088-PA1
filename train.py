"""
    [AUE8088] PA1: Image Classification
        - To run: (aue8088) $ python train.py
        - For better flexibility, consider using LightningCLI in PyTorch Lightning
"""
# train.py (수정)
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning import Trainer
import torch

from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
import src.config as cfg

torch.set_float32_matmul_precision('medium')

def run_experiment(optimizer_params, scheduler_params, run_name):
    # 모델 & 데이터셋
    model = SimpleClassifier(
        model_name = cfg.MODEL_NAME,
        num_classes = cfg.NUM_CLASSES,
        optimizer_params = optimizer_params,
        scheduler_params = scheduler_params,
    )
    datamodule = TinyImageNetDatasetModule(batch_size = cfg.BATCH_SIZE)

    # Wandb logger (실험별 이름 지정)
    wandb_logger = WandbLogger(
        project = cfg.WANDB_PROJECT,
        save_dir = cfg.WANDB_SAVE_DIR,
        entity = cfg.WANDB_ENTITY,
        name = run_name,
        group    = "sweep_hParams",
    )
    # ⚠️ 실제 사용된 값들을 wandb에 업데이트
    wandb_logger.experiment.config.update({
        'lr': optimizer_params['lr'],
        'momentum': optimizer_params['momentum'],
        'scheduler_milestones': scheduler_params['milestones'],
        'scheduler_gamma': scheduler_params['gamma'],
    })

    # Trainer 설정
    trainer = Trainer(
        accelerator = cfg.ACCELERATOR,
        devices = cfg.DEVICES,
        precision = cfg.PRECISION_STR,
        max_epochs = cfg.NUM_EPOCHS,
        check_val_every_n_epoch = cfg.VAL_EVERY_N_EPOCH,
        logger = wandb_logger,
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(save_top_k=1, monitor='f1/val', mode='max'),
        ],
    )

    # 학습 & 검증
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(ckpt_path='best', datamodule=datamodule)

    # finish wandb logging
    wandb_logger.experiment.finish()

if __name__ == "__main__":
    # 1) Learning‐rate sweep (0.005, 0.01, 0.05) → 총 3실험
    for lr in cfg.LR_LIST:
        opt = dict(cfg.OPTIMIZER_PARAMS)
        opt['lr'] = lr
        run_experiment(opt, cfg.SCHEDULER_PARAMS, f"{cfg.MODEL_NAME}-lr{lr}")

    # 2) Momentum sweep (0.9, 0.1, 1.5) → 총 3실험
    for m in cfg.MOMENTUM_LIST:
        opt = dict(cfg.OPTIMIZER_PARAMS)
        opt['momentum'] = m
        run_experiment(opt, cfg.SCHEDULER_PARAMS, f"{cfg.MODEL_NAME}-mom{m}")

    # 3) Scheduler milestones sweep ([30,35], [20,30]) → 총 2실험
    for sched in cfg.SCHEDULER_LIST:
        name = f"{cfg.MODEL_NAME}-ms{'-'.join(map(str, sched['milestones']))}"
        run_experiment(cfg.OPTIMIZER_PARAMS, sched, name)