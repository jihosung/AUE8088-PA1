# Python packages
from termcolor import colored
from typing import Dict
import copy
import pandas as pd
import os

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch

# Custom packages
from src.metric import MyAccuracy
from src.metric import MyF1Score
import src.config as cfg
from src.util import show_setting


# [TODO: Optional] Rewrite this class if you want
class MyNetwork(AlexNet):
    def __init__(self):
        super().__init__()

        # [TODO] Modify feature extractor part in AlexNet


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [TODO: Optional] Modify this as well if you want
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        if model_name == 'MyNetwork':
            self.model = MyNetwork()
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.accuracy = MyAccuracy()
        self.f1score = MyF1Score()

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1 = self.f1score(scores, y)
        self.log_dict({
            'loss/train': loss,
            'accuracy/train': accuracy,
            'f1/train': f1,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1 = self.f1score(scores, y)
        self.log_dict({
            'loss/val': loss,
            'accuracy/val': accuracy,
            'f1/val': f1,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency=cfg.WANDB_IMG_LOG_FREQ)


    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])

    def on_validation_epoch_end(self):
        # check valid state
        if not (hasattr(self.f1score, 'f1_per_class') and isinstance(self.logger, WandbLogger)):
            return

        # 1. 저장 디렉토리 생성 (wandb run 이름 기반)
        run_name = self.logger.experiment.name  # wandb run 이름
        save_dir = os.path.join("f1_logs", run_name)
        os.makedirs(save_dir, exist_ok=True)

        # 2. per-class F1 가져오기 & 저장
        f1_values = self.f1score.f1_per_class.cpu().numpy()
        df = pd.DataFrame({
            'class': list(range(self.f1score.num_classes)),
            'f1_score': f1_values
        })
        df.to_csv(os.path.join(save_dir, f'f1_scores_epoch_{self.current_epoch:03d}.csv'), index=False)

        # 3. Top-5 / Bottom-5 클래스 WandB에 로그
        topk = torch.topk(self.f1score.f1_per_class, 5)
        bottomk = torch.topk(-self.f1score.f1_per_class, 5)

        log_dict = {
            f'f1/top_class_{i}': f1_values[idx] for i, idx in enumerate(topk.indices)
        } | {
            f'f1/bottom_class_{i}': f1_values[idx] for i, idx in enumerate(bottomk.indices)
        } | {
            'epoch': self.current_epoch
        }
        self.logger.experiment.log(log_dict)
