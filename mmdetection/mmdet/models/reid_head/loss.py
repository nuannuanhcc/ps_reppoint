# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import numpy as np


def circle_loss(
    sim_ap: torch.Tensor,
    sim_an: torch.Tensor,
    scale: float = 16.0,
    margin: float = 0.4,
    redection: str = "mean"
):
    pair_ap = -scale * (sim_ap - margin)
    pair_an = scale * sim_an
    pair_ap = torch.logsumexp(pair_ap, dim=1)
    pair_an = torch.logsumexp(pair_an, dim=1)
    loss = torch.nn.functional.softplus(pair_ap + pair_an)
    if redection == "mean":
        loss = loss.mean()
    elif redection == "sum":
        loss = loss.sum()
    return loss

@torch.no_grad()
def update_queue(queue, pointer, new_item):
    n = new_item.shape[0]
    length = queue.shape[0]
    if pointer + n <= length:
        queue[pointer: pointer + n] = new_item
        pointer = pointer + n
    else:
        res = n-(length-pointer)
        queue[pointer: length] = new_item[:length-pointer]
        queue[: res] = new_item[-res:]
        pointer = res
    return queue, pointer

@torch.no_grad()
def update_lut(lut, feat, id, m):
    for x, y in zip(feat, id):
        lut[y] = m * lut[y] + (1 - m) * x
        lut[y] = F.normalize(lut[y], dim=-1)
    return lut

class OIM(Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut, queue, num_gt, momentum):
        ctx.lut = lut
        ctx.queue = queue
        ctx.num_gt = num_gt
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs_labeled = inputs.mm(ctx.lut.t())
        outputs_unlabeled = inputs.mm(ctx.queue.t())
        return torch.cat((outputs_labeled, outputs_unlabeled), 1)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_outputs, = grad_outputs
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat((ctx.lut, ctx.queue), 0))

        for i, (x, y) in enumerate(zip(inputs, targets)):
            if y == -1:
                tmp = torch.cat((ctx.queue[1:], x.view(1, -1)), 0)
                ctx.queue[:, :] = tmp[:, :]
            elif 0 <= y < len(ctx.lut):
                if i < ctx.num_gt:
                    ctx.lut[y] = ctx.momentum * ctx.lut[y] + (1. - ctx.momentum) * x
                    ctx.lut[y] = F.normalize(ctx.lut[y], dim=-1)
            else:
                continue
        return grad_inputs, None, None, None, None, None


class OIMLossComputation(nn.Module):
    def __init__(self, cfg):
        super(OIMLossComputation, self).__init__()
        self.cfg = cfg
        if self.cfg.dataset_type == 'SysuDataset':
            self.num_pid = 5532
            self.queue_size = 5000
        elif self.cfg.dataset_type == 'PrwDataset':
            self.num_pid = 483
            self.queue_size = 500
        else:
            raise KeyError(cfg.DATASETS.TRAIN)

        self.lut_momentum = 0.0
        self.out_channels = 2048

        self.register_buffer('lut', torch.zeros(self.num_pid, self.out_channels).cuda())
        self.register_buffer('queue', torch.zeros(self.queue_size, self.out_channels).cuda())

    def forward(self, features, gt_labels):

        pids = torch.cat([i[:, -1] for i in gt_labels])
        aux_label = pids  # threshold<0.7 pid=-2

        pid_label = torch.from_numpy(pids.cpu().numpy()).long().cuda().view(-1)  # threshold<0.7 pid=-1

        num_gt = pids.shape[0]

        reid_result = OIM.apply(features, aux_label, self.lut, self.queue, num_gt, self.lut_momentum)
        loss_weight = torch.cat([torch.ones(self.num_pid).cuda(), torch.zeros(self.queue_size).cuda()])

        scalar = 10
        loss_reid = F.cross_entropy(reid_result * scalar, pid_label, weight=loss_weight, ignore_index=-1)
        return loss_reid


class CIRCLELossComputation(nn.Module):
    def __init__(self, cfg):
        super(CIRCLELossComputation, self).__init__()
        self.cfg = cfg

        if self.cfg.dataset_type == 'SysuDataset':
            num_labeled = 15080
            num_unlabeled = 8192
        elif self.cfg.dataset_type == 'PrwDataset':
            num_labeled = 8192
            num_unlabeled = 8192
        else:
            raise KeyError(cfg.DATASETS.TRAIN)

        self.m = 0.5
        self.out_channels = 2048

        self.register_buffer('labels', torch.arange(num_labeled, dtype=torch.long).cuda())
        self.register_buffer('features', torch.zeros(num_labeled, self.out_channels).cuda())

    def forward(self, features, features_k, gt_labels, gt_labels_k):
        pids = torch.cat([i[:, -1] for i in gt_labels])
        id_labeled = pids[pids > -1]
        feat_labeled = features[pids > -1]

        pids_k = torch.cat([i[:, -1] for i in gt_labels_k])
        id_labeled_k = pids_k[pids_k > -1]
        feat_labeled_k = features_k[pids_k > -1]

        if not id_labeled.numel():
            return torch.tensor(0.0)

        self.features = update_lut(self.features, feat_labeled_k, id_labeled_k, self.m)
        pseudo_id_labeled = self.labels[id_labeled]

        lut_sim = torch.mm(feat_labeled, self.features.t())
        positive_mask = pseudo_id_labeled.view(-1, 1) == self.labels.view(1, -1)
        sim_ap = lut_sim.masked_fill(~positive_mask, float("inf"))
        sim_an = lut_sim.masked_fill(positive_mask, float("-inf"))
        pair_loss = circle_loss(sim_ap, sim_an)
        return pair_loss

class CIRCLELoss_Cluster(nn.Module):
    def __init__(self, cfg):
        super(CIRCLELoss_Cluster, self).__init__()
        self.cfg = cfg
        self.m = 0.5
        if self.cfg.dataset_type == 'SysuDataset':
            num_labeled = 55260
        elif self.cfg.dataset_type == 'PrwDataset':
            num_labeled = 8192
        else:
            raise KeyError(cfg.DATASETS.TRAIN)

        self.out_channels = 2048

        self.register_buffer('labels',   torch.arange(num_labeled, dtype=torch.long).cuda())
        self.register_buffer('features', torch.zeros(num_labeled, self.out_channels).cuda())

    def forward(self, features, features_k, gt_labels, gt_labels_k):
        pids = torch.cat([i[:, -1] for i in gt_labels])
        pseudo_pid = torch.cat([self.labels[i[:, -1]] for i in gt_labels])
        self.features= update_lut(self.features, features_k, pids, self.m)
        queue_sim = torch.mm(features, self.features.t())
        positive_mask = pseudo_pid.view(-1, 1) == self.labels.view(1, -1)
        sim_ap = queue_sim.masked_fill(~positive_mask, float("inf"))
        sim_an = queue_sim.masked_fill(positive_mask, float("-inf"))
        pair_loss = circle_loss(sim_ap, sim_an)
        return pair_loss


def make_reid_loss_evaluator(cfg):
    # loss_evaluator = OIMLossComputation(cfg)
    # loss_evaluator = CIRCLELossComputation(cfg)
    loss_evaluator = CIRCLELossComputation(cfg)
    return loss_evaluator
