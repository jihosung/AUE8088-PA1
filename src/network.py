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


class MyNetwork(AlexNet):
    def __init__(self):
        super().__init__()
        # [TODO] Modify feature extractor part in AlexNet if needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SimpleClassifier(LightningModule):
    def __init__(
        self,
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
            assert model_name in models_list, (
                f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            )
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics: separate instances for train and val
        self.train_f1 = MyF1Score(num_classes=num_classes)
        self.val_f1 = MyF1Score(num_classes=num_classes)
        self.accuracy = MyAccuracy()  # can reuse for both if stateless per-step

        # Buffers for prediction distribution debug
        self._val_preds = []
        self._val_targets = []

        # Hyperparameters
        self.save_hyperparameters()

        # count # of params
        total_params = sum(p.numel() for p in self.model.parameters())
        self.hparams.total_params = total_params

    def on_train_start(self):
        show_setting(cfg)

    def on_fit_start(self):
        self.logger.experiment.log({'total_params': self.hparams.total_params})

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
        acc = self.accuracy(scores, y)
        self.train_f1.update(scores, y)
        self.log_dict(
            {
                'loss/train': loss,
                'accuracy/train': acc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def on_train_epoch_end(self):
        # compute and log train F1
        train_f1_val = self.train_f1.compute()
        self.log(
            'f1/train',
            train_f1_val,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        # reset for next epoch
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        acc = self.accuracy(scores, y)
        self.val_f1.update(scores, y)

        preds = torch.argmax(scores, dim=1)
        self._val_preds.append(preds.cpu())
        self._val_targets.append(y.cpu())

        self.log_dict(
            {
                'loss/val': loss,
                'accuracy/val': acc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        self._wandb_log_image(batch, batch_idx, scores, frequency=cfg.WANDB_IMG_LOG_FREQ)

    def on_validation_epoch_start(self):
        self._val_preds = []
        self._val_targets = []
        # reset F1 metric state
        self.val_f1.reset()

    def on_validation_epoch_end(self):
        val_f1_val = self.val_f1.compute()
        # log overall epoch-level validation F1 for Lightning and ModelCheckpoint
        self.log(
            'f1/val',
            val_f1_val,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        # save per-class F1 to CSV
        if isinstance(self.logger, WandbLogger):
            run_name = self.logger.experiment.name
            save_dir = os.path.join("f1_logs", run_name)
            os.makedirs(save_dir, exist_ok=True)

            f1_values = self.val_f1.f1_per_class.cpu().numpy()
            df = pd.DataFrame({'class': list(range(self.val_f1.num_classes)), 'f1_score': f1_values})
            df.to_csv(
                os.path.join(save_dir, f'f1_scores_epoch_{self.current_epoch:03d}.csv'),
                index=False
            )

            # log top-5 / bottom-5 classes via Lightning
            topk = torch.topk(self.val_f1.f1_per_class, 5)
            bottomk = torch.topk(-self.val_f1.f1_per_class, 5)
            top_dict = {f'f1/top_class_{i}': f1_values[idx] for i, idx in enumerate(topk.indices)}
            bottom_dict = {f'f1/bottom_class_{i}': f1_values[idx] for i, idx in enumerate(bottomk.indices)}
            self.log_dict({**top_dict, **bottom_dict}, on_step=False, on_epoch=True, logger=True)

        # debug: print prediction vs target distribution
        all_preds = torch.cat(self._val_preds)
        unique_preds, counts = torch.unique(all_preds, return_counts=True)
        print("\n[Validation Prediction Distribution]")
        for cls, cnt in zip(unique_preds.tolist(), counts.tolist()):
            print(f"Class {cls:3d}: {cnt} times")

        all_targets = torch.cat(self._val_targets)
        unique_targets, tcounts = torch.unique(all_targets, return_counts=True)
        print("\n[Validation Target Distribution]")
        for cls, cnt in zip(unique_targets.tolist(), tcounts.tolist()):
            print(f"Class {cls:3d}: {cnt} times")

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency=100):
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
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}']
            )
