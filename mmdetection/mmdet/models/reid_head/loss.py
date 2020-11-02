# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import numpy as np
from torch import nn, autograd

def circle_loss(
    sim_ap: torch.Tensor,
    sim_an: torch.Tensor,
    scale: float = 16.0,
    margin: float = 0.0,
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
            num_labeled = 8192
            num_unlabeled = 8192
        elif self.cfg.dataset_type == 'PrwDataset':
            num_labeled = 8192
            num_unlabeled = 8192
        else:
            raise KeyError(cfg.DATASETS.TRAIN)

        self.out_channels = 2048

        self.register_buffer('pointer', torch.zeros(2, dtype=torch.int).cuda())
        self.register_buffer('id_inx', -torch.ones(num_labeled, dtype=torch.long).cuda())
        self.register_buffer('lut', torch.zeros(num_labeled, self.out_channels).cuda())
        self.register_buffer('queue', torch.zeros(num_unlabeled, self.out_channels).cuda())

    def forward(self, features, gt_labels):

        pids = torch.cat([i[:, -1] for i in gt_labels])
        aux_label = pids  # threshold<0.7 pid=-2

        aux_label_np = aux_label.data.cpu().numpy()
        invalid_inds = np.where((aux_label_np < 0))
        aux_label_np[invalid_inds] = -1

        id_labeled = aux_label[aux_label > -1].to(torch.long)
        feat_labeled = features[aux_label > -1]
        feat_unlabeled = features[aux_label == -1]
        self.lut, _ = update_queue(self.lut, self.pointer[0], feat_labeled)

        self.id_inx, self.pointer[0] = update_queue(self.id_inx, self.pointer[0], id_labeled)
        self.queue, self.pointer[1] = update_queue(self.queue, self.pointer[1], feat_unlabeled)

        if not id_labeled.numel():
            return torch.tensor(0.0)

        queue_sim = torch.mm(feat_labeled, self.queue.t())
        lut_sim = torch.mm(feat_labeled, self.lut.t())
        positive_mask = id_labeled.view(-1, 1) == self.id_inx.view(1, -1)
        sim_ap = lut_sim.masked_fill(~positive_mask, float("inf"))
        sim_an = lut_sim.masked_fill(positive_mask, float("-inf"))
        sim_an = torch.cat((queue_sim, sim_an), dim=-1)

        pair_loss = circle_loss(sim_ap, sim_an)
        return pair_loss


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self,cfg, num_features=2048, num_samples=55260, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.cfg = cfg
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, gt_labels):
        # inputs: B*2048, features: L*2048

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)

        pids = torch.cat([i[:, -1] for i in gt_labels])
        similarities = hm(inputs, pids, self.features, self.momentum)
        similarities /= self.temp
        B = similarities.size(0)

        targets = torch.cat([self.labels[i[:, -1]] for i in gt_labels])
        labels = self.labels.clone()

        sim = torch.zeros(labels.max()+1, B).float().cuda()
        sim.index_add_(0, labels, similarities.t().contiguous())   # 相同label的累加在一起
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples,1).float().cuda())
        mask = (nums>0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)  #得到sim的平均值
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        return F.nll_loss(torch.log(masked_sim+1e-6), targets)


def make_reid_loss_evaluator(cfg):
    # loss_evaluator = OIMLossComputation(cfg)
    # loss_evaluator = CIRCLELossComputation(cfg)
    loss_evaluator = HybridMemory(cfg)
    return loss_evaluator
