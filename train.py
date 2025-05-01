# train.py
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning import Trainer
import torch

from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
import src.config as cfg

torch.set_float32_matmul_precision('medium')

def run_experiment(model_name, optimizer_params, scheduler_params, run_name):
    # 모델 & 데이터셋 초기화
    model = SimpleClassifier(
        model_name       = model_name,
        num_classes      = cfg.NUM_CLASSES,
        optimizer_params = optimizer_params,
        scheduler_params = scheduler_params,
    )
    datamodule = TinyImageNetDatasetModule(batch_size=cfg.BATCH_SIZE)

    # WandB 로거 설정 (group 추가)
    wandb_logger = WandbLogger(
        project = cfg.WANDB_PROJECT,
        save_dir = cfg.WANDB_SAVE_DIR,
        entity   = cfg.WANDB_ENTITY,
        name     = run_name,
        group    = "sweepModel",
    )
    wandb_logger.experiment.config.update({
        'model_name': model_name,
        **optimizer_params,
        **scheduler_params,
    })

    # Trainer 설정
    trainer = Trainer(
        accelerator               = cfg.ACCELERATOR,
        devices                   = cfg.DEVICES,
        precision                 = cfg.PRECISION_STR,
        max_epochs                = cfg.NUM_EPOCHS,
        check_val_every_n_epoch   = cfg.VAL_EVERY_N_EPOCH,
        logger                    = wandb_logger,
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(save_top_k=1, monitor='f1/val', mode='max'),
        ],
    )

    # 학습 및 검증
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(ckpt_path='best', datamodule=datamodule)

    # WandB 세션 종료
    wandb_logger.experiment.finish()

if __name__ == "__main__":
    # 모델 리스트만 순회하며 실행
    for model_name in cfg.MODEL_LIST:
        run_experiment(
            model_name       = model_name,
            optimizer_params = cfg.OPTIMIZER_PARAMS,
            scheduler_params = cfg.SCHEDULER_PARAMS,
            run_name         = model_name
        )
