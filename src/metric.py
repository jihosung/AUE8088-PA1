from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__() # parent class init
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx='sum') # add tp state
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx='sum') # add fp state
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx='sum') # add fn state

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)

        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch: {preds.shape} vs {target.shape}")
        
        self.tp += ((preds == 1) & (target == 1)).sum()
        self.fp += ((preds == 1) & (target == 0)).sum()
        self.fn += ((preds == 0) & (target == 1)).sum()

    def compute(self):
        precision = self.tp.float() / (self.tp + self.fp).float().clamp(min=1e-10)
        recall = self.tp.float() / (self.tp + self.fn).float().clamp(min=1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1

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
