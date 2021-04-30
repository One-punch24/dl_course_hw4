import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):

    def __init__(self, args):
        super().__init__()
        # TODO

    def logits(self, source_tokens, segment_ids, **unused):
        # TODO
    
    def get_loss(self, source_tokens, segment_ids, targets, **unused):
        logits = self.logits(source_tokens, segment_ids)
        loss = F.cross_entropy(logits, targets)
        return loss
