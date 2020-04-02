import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import LSTMCell
from JSL.verb.resnet50 import ResNet
import pdb
import torch.utils.model_zoo as model_zoo



class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
    def forward(self, x, target):

        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class ImsituVerb(nn.Module):
    def __init__(self):
        super(ImsituVerb, self).__init__()

        self.feature_extractor = ResNet()
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', model_dir='.')
        self.feature_extractor.load_state_dict(state_dict, strict=False)
        self.verb_linear = nn.Linear(2048, 504)
        self.loss_function = LabelSmoothing(0.2)


    def forward(self, epoch_num, image, verb, is_train=True):

        pred = self.feature_extractor(image, epoch_num)
        verb_pred = self.verb_linear(pred)

        if is_train:
            return self.loss_function(verb_pred, verb.long().squeeze()), torch.argmax(verb_pred, dim=1)
        else:
            return torch.argmax(verb_pred, dim=1), torch.topk(verb_pred, 5, dim=1)[1]
