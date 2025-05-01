from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes=200):
        super().__init__()
        self.num_classes = num_classes
        self.add_state('tp', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.f1_per_class = None

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        for c in range(self.num_classes):
            pred_c = preds == c
            target_c = target == c
            self.tp[c] += (pred_c & target_c).sum()
            self.fp[c] += (pred_c & ~target_c).sum()
            self.fn[c] += (~pred_c & target_c).sum()

    def compute(self):
        precision = self.tp / (self.tp + self.fp).clamp(min=1e-10)
        recall = self.tp / (self.tp + self.fn).clamp(min=1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        self.f1_per_class = f1.detach().cpu()
        return f1.mean()


class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise ValueError(f"Predictions and targets must have the same shape, got {preds.shape} and {target.shape}")

        # [TODO] Cound the number of correct prediction
        correct = (preds == target).sum()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
