"""Training-only, top-k false-alarm penalty for original empty-label images.

An empty annotation is a candidate-normal label, not human certification.
Native YOLO detection losses and inference/checkpoint model classes are unchanged.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss


def hard_normal_penalty(logits, eligible, topk):
    """B,C,A raw logits -> batch-mean BCE on each normal image's top-k locations.

    Each location contributes its maximum class logit once. Averaging over the
    whole local batch (not only normal images) keeps auxiliary scaling consistent
    across equal-sized DDP workers, even with unequal normal-image counts.
    """
    if logits.ndim != 3 or eligible.shape != (logits.shape[0],) or eligible.dtype != torch.bool:
        raise ValueError("Expected B,C,A logits and a boolean eligibility vector of length B")
    if type(topk) is not int or not 1 <= topk <= logits.shape[2]:
        raise ValueError("topk must fit the number of prediction locations")
    if not eligible.any():
        return logits.reshape(-1)[0].float() * 0.0
    hardest = logits[eligible].float().amax(dim=1).topk(topk, dim=1).values
    return F.softplus(hardest).mean(dim=1).sum() / logits.shape[0]


class HardNormalLoss(v8DetectionLoss):
    def __init__(self, model, weight, topk, normal_paths):
        super().__init__(model)
        self.weight, self.topk = float(weight), topk
        self.normal_paths = frozenset(str(Path(p).resolve()) for p in normal_paths)
        self.observed_batches = self.observed_images = self.eligible_images = 0
        self.auxiliary_sum = self.native_sum = 0.0

    def loss(self, preds, batch):
        native, items = super().loss(preds, batch)
        logits = preds["scores"]  # pinned 8.4.115: B,C,A, before sigmoid and NMS
        if logits.shape[1] != self.nc or len(batch["im_file"]) != logits.shape[0]:
            raise ValueError("Unexpected score layout or missing per-image provenance")
        eligible = torch.tensor([str(Path(p).resolve()) in self.normal_paths for p in batch["im_file"]],
                                dtype=torch.bool, device=logits.device)
        # Original positives cropped to zero boxes are deliberately ineligible.
        occupied = batch["batch_idx"].long().unique().to(logits.device)
        if eligible[occupied].any():
            raise RuntimeError("An originally normal image acquired boxes; check mixed-image augmentation")
        extra = self.weight * hard_normal_penalty(logits, eligible, self.topk)
        self.observed_batches += 1
        self.observed_images += logits.shape[0]
        self.eligible_images += int(eligible.sum().detach())
        self.auxiliary_sum += float(extra.detach())
        self.native_sum += float(native.detach().sum()) / logits.shape[0]
        # Native loss components are batch-multiplied; use the same convention.
        return torch.cat((native, (extra * logits.shape[0]).view(1))), {**items, "hard_normal_loss": extra.detach()}

    def summary(self):
        return {"batches": self.observed_batches, "images": self.observed_images,
                "eligible_presentations": self.eligible_images,
                "sum_batch_mean_auxiliary": self.auxiliary_sum,
                "sum_batch_mean_native": self.native_sum,
                "auxiliary_to_native_loss_ratio": self.auxiliary_sum / self.native_sum if self.native_sum else None}


def install_hard_normal(trainer, settings):
    """Install after the loader exists; keep EMA validation on native loss."""
    from ultralytics.utils.torch_utils import unwrap_model
    model = unwrap_model(trainer.model)
    if model.end2end or type(model.init_criterion()) is not v8DetectionLoss:
        raise ValueError("This extension is qualified only for native YOLOv8 detection")
    if any(getattr(trainer.args, key) != 0 for key in ("mosaic", "mixup", "cutmix", "copy_paste")):
        raise ValueError("Effective mixed-image augmentation must be disabled")
    labels = trainer.train_loader.dataset.labels
    normal_paths = [item["im_file"] for item in labels if len(item["cls"]) == 0]
    if not normal_paths:
        raise ValueError("No original empty-label images are available for the extra loss")
    criterion = HardNormalLoss(model, settings["weight"], settings["topk"], normal_paths)
    model.criterion = criterion
    return criterion
