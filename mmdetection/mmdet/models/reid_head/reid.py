import torch
import torch.nn.functional as F
from torch import nn

# from .reid_feature_extractors import make_reid_feature_extractor
from .loss import make_reid_loss_evaluator

class REIDModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and outputs
    RPN proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(REIDModule, self).__init__()

        self.cfg = cfg

        # self.feature_extractor = make_reid_feature_extractor(cfg)
        # self.loss_evaluator = make_reid_loss_evaluator(cfg)
        # self.fc1 = nn.Linear(256*7*7, 1024)
        # self.fc2 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(256 * 7 * 7, 2048)

    def forward(self, x, gt_labels=None):

        x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x), inplace=False)
        x = self.fc2(x)
        feats = F.normalize(x, dim=-1)
        return feats
        # loss_reid = self.loss_evaluator(feats, gt_labels)
        # return {"loss_reid": [loss_reid], }


def build_reid(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return REIDModule(cfg)