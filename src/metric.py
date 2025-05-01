from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes=200):
        super().__init__()
        self.num_classes = num_classes

        # 클래스별 TP, FP, FN 누적용 상태 정의
        self.add_state('tp', default=torch.zeros(num_classes, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros(num_classes, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros(num_classes, dtype=torch.float), dist_reduce_fx='sum')

        self.f1_per_class = None  # on_validation_epoch_end용

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # 예측 결과 argmax 처리
        preds = torch.argmax(preds, dim=1)

        # Shape 체크 (안 맞으면 에러 발생)
        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch: preds.shape={preds.shape}, target.shape={target.shape}")

        # 클래스별 TP, FP, FN 계산 및 누적
        for c in range(self.num_classes):
            pred_c = preds == c
            target_c = target == c

            self.tp[c] += torch.sum(pred_c & target_c).float()
            self.fp[c] += torch.sum(pred_c & ~target_c).float()
            self.fn[c] += torch.sum(~pred_c & target_c).float()

    def compute(self):
        # print("TP:", self.tp)
        # print("FP:", self.fp)
        # print("FN:", self.fn)

        # Precision, Recall 계산
        precision = self.tp / (self.tp + self.fp).clamp(min=1e-10)
        recall = self.tp / (self.tp + self.fn).clamp(min=1e-10)

        # F1-score 계산 (클래스별)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        # 외부에서 접근할 수 있도록 저장
        self.f1_per_class = f1.detach().cpu()

        # 평균 F1-score 반환
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