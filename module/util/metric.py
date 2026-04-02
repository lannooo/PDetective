import torch
from torch import Tensor
from torchmetrics.classification import ROC, PrecisionRecallCurve, AUROC

class BasicMetric:
    def _calcualte_eer(self, roc:ROC):
        fpr, tpr, thresholds = roc.compute()
        fnr = 1 - tpr
        # calculate EER and threshold based on the FPR and FNR
        idx = torch.argmin(torch.absolute(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2
        threshold = thresholds[idx]
        return eer, threshold
    
    def _calculate_precision_recall(self, prc:PrecisionRecallCurve):
        precision, recall, thresholds = prc.compute()
        assert len(precision) == len(recall) == len(thresholds)+1, f"Precision ({precision.shape}), recall ({recall.shape}), and thresholds ({thresholds.shape}) must have the same length."
        # calcualte a best threshold
        f1 = 2 * (precision * recall) / (precision + recall)
        # there maybe are nan values, replace nan to zero
        f1[torch.isnan(f1)] = 0
        # get the max one index
        idx = torch.argmax(f1)
        print("Calculating metric index", idx)
        # assert idx.size(0) == 1, f"Expected a single index, but got {idx}: {f1}"
        return precision[idx], recall[idx], thresholds[idx]
    
    def _calculate_auroc(self, auroc:AUROC):
        return auroc.compute()
    
    def report(self, stage):
        raise NotImplementedError("Please implement the report method in the subclass.")
    
    def reset(self, stage=None):
        raise NotImplementedError("Please implement the reset method in the subclass.")


class BoundaryMetric(BasicMetric):
    def __init__(self):
        super().__init__()
        self.roc_dict = {}
        for k in ['train', 'validate', 'test']:
            self.roc_dict[k] = ROC(task='binary')

    def reset(self, stage=None):
        keys = [stage] if stage is not None else ['train', 'validate', 'test']
        for k in keys:
            self.roc_dict[k].reset()

    def update(self, stage, b_preds: Tensor, b_target: Tensor, b_lengths):
        assert stage in ['train', 'validate', 'test'], f"Invalid stage: {stage}"
        for i, n_frame in enumerate(b_lengths):
            preds = b_preds[i, :n_frame].flatten()
            target = b_target[i, :n_frame].flatten()
            self.roc_dict[stage].update(preds, target)

    def report(self, stage):
        assert stage in ['train', 'validate', 'test'], f"Invalid stage: {stage}"
        eer, thres1 = self._calcualte_eer(self.roc_dict[stage])
        return eer, thres1


class FrameMetric(BasicMetric):
    def __init__(self):
        super().__init__()
        self.prc_dict = {}
        self.roc_dict = {}
        self.auroc_dict = {}
        for k in ['train', 'validate', 'test']:
            self.prc_dict[k] = PrecisionRecallCurve(task='binary')
            self.roc_dict[k] = ROC(task='binary')
            self.auroc_dict[k] = AUROC(task='binary')
    
    def reset(self, stage=None):
        keys = [stage] if stage is not None else ['train', 'validate', 'test']
        for k in keys:
            self.prc_dict[k].reset()
            self.roc_dict[k].reset()
            self.auroc_dict[k].reset()

    def update(self, stage, frame_preds: Tensor, frame_target: Tensor, frame_lengths):
        assert stage in ['train', 'validate', 'test'], f"Invalid stage: {stage}"
        for i, n_frame in enumerate(frame_lengths):
            preds = frame_preds[i, :n_frame].flatten()
            target = frame_target[i, :n_frame].flatten()
            self.prc_dict[stage].update(preds, target)
            self.roc_dict[stage].update(preds, target)
            self.auroc_dict[stage].update(preds, target)
    
    def update_(self, stage, frame_preds: Tensor, frame_target: Tensor):
        assert stage in ['train', 'validate', 'test'], f"Invalid stage: {stage}"
        if frame_preds.dim() > 2:
            frame_preds = frame_preds.flatten()
        if frame_target.dim() > 2:
            frame_target = frame_target.flatten()
        self.prc_dict[stage].update(frame_preds, frame_target)
        self.roc_dict[stage].update(frame_preds, frame_target)
        self.auroc_dict[stage].update(frame_preds, frame_target)
    
    def report(self, stage):
        assert stage in ['train', 'validate', 'test'], f"Invalid stage: {stage}"
        eer, thres1 = self._calcualte_eer(self.roc_dict[stage])
        precision, recall, thres2 = self._calculate_precision_recall(self.prc_dict[stage])
        auroc = self._calculate_auroc(self.auroc_dict[stage])
        return eer, thres1, precision, recall, thres2, auroc

class UtteranceMetric(BasicMetric):
    def __init__(self):
        super().__init__()
        self.prc_dict = {}
        self.roc_dict = {}
        self.auroc_dict = {}
        for k in ['train', 'validate', 'test']:
            self.prc_dict[k] = PrecisionRecallCurve(task='binary')
            self.roc_dict[k] = ROC(task='binary')
            self.auroc_dict[k] = AUROC(task='binary')

    def reset(self, stage=None):
        keys = [stage] if stage is not None else ['train', 'validate', 'test']
        for k in keys:
            self.prc_dict[k].reset()
            self.roc_dict[k].reset()
            self.auroc_dict[k].reset()
    
    def update(self, stage, preds: Tensor, target: Tensor):
        assert stage in ['train', 'validate', 'test'], f"Invalid stage: {stage}"
        if preds.dim() > 2:
            preds = preds.flatten()
        if target.dim() > 2:
            target = target.flatten()
        self.prc_dict[stage].update(preds, target)
        self.roc_dict[stage].update(preds, target)
        self.auroc_dict[stage].update(preds, target)
    
    def report(self, stage):
        assert stage in ['train', 'validate', 'test'], f"Invalid stage: {stage}"
        eer, thres1 = self._calcualte_eer(self.roc_dict[stage])
        precision, recall, thres2 = self._calculate_precision_recall(self.prc_dict[stage])
        auroc = self._calculate_auroc(self.auroc_dict[stage])
        return eer, thres1, precision, recall, thres2, auroc