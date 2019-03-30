import numpy as np
import torch


def accuracy(outputs: torch.Tensor, labels: torch.Tensor, ignore_index: int=None) -> float:
    # Num of class should be less than 255.

    if len(outputs.shape) == 4:
        preds = outputs.argmax(dim=1)
    elif len(outputs.shape) == 3:
        preds = outputs
    else:
        raise ValueError

    preds = preds.byte().flatten()
    labels = labels.byte().flatten()

    if ignore_index is not None:
        is_not_ignore = labels != ignore_index
        preds = preds[is_not_ignore]
        labels = labels[is_not_ignore]

    correct = preds.eq(labels)

    acc = correct.float().mean().item()

    return acc


def prec_at_k(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    if len(res) == 1:
        res = res[0]

    return res


def intersection_and_union(preds: torch.Tensor, labels: torch.Tensor,
                           ignore_index=255, n_classes=19):

    assert ignore_index > n_classes, 'ignore_index should be grater than n_classes'

    preds = preds.byte().flatten()
    labels = labels.byte().flatten()

    is_not_ignore = labels != ignore_index
    preds = preds[is_not_ignore]
    labels = labels[is_not_ignore]

    intersection = preds[preds == labels]
    area_intersection = intersection.bincount(minlength=n_classes)

    bincount_preds = preds.bincount(minlength=n_classes)
    bincount_labels = labels.bincount(minlength=n_classes)
    area_union = bincount_preds + bincount_labels - area_intersection

    area_intersection = area_intersection.float().cpu().numpy()
    area_union = area_union.float().cpu().numpy()

    return area_intersection, area_union


def mean_iou(outputs, labels, n_classes=19):

    preds = outputs.argmax(dim=1)
    intersection, union = intersection_and_union(preds, labels, n_classes=n_classes)

    return np.mean(intersection / (union + 1e-16))


def mean_iou_50_to_95(outputs: torch.Tensor, labels: torch.Tensor,
                      thresh=None, eps=1e-7, reduce=True):

    if thresh is not None:
        outputs = outputs > thresh

    outputs = outputs.squeeze(1)
    labels = labels.squeeze(1).byte()

    intersection = (outputs & labels).sum(dim=[1, 2]).float()
    union = (outputs | labels).sum(dim=[1, 2]).float()

    iou = (intersection + eps) / (union + eps)
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

    if reduce:
        thresholded = thresholded.mean()

    return thresholded
