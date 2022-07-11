import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class OutOpenEnded(nn.Module):
    def __init__(self, embed_dim, num_answers, drorate, activation):
        super(OutOpenEnded, self).__init__()
        if activation == 'relu':
            self.activ = nn.ReLU()
        if activation == 'prelu':
            self.activ = nn.PReLU()
        if activation == 'elu':
            self.activ = nn.ELU()
        if activation == 'gelu':
            self.activ = nn.GELU()
        self.classifier = nn.Sequential(
                                        self.activ,
                                        nn.LayerNorm(embed_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(embed_dim, num_answers))


    def forward(self, x):
        out = self.classifier(x)
        return out


class OutCount(nn.Module):
    def __init__(self, embed_dim, drorate, activation):
        super(OutCount, self).__init__()
        if activation == 'relu':
            self.activ = nn.ReLU()
        if activation == 'prelu':
            self.activ = nn.PReLU()
        if activation == 'elu':
            self.activ = nn.ELU()
        if activation == 'gelu':
            self.activ = nn.GELU()

        self.regression = nn.Sequential(
                                        self.activ,
                                        nn.LayerNorm(embed_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(embed_dim, 1))


    def forward(self, x):
        out = self.regression(x)
        return out


class OutMultiChoices(nn.Module):
    def __init__(self, embed_dim, drorate, activation='gelu'):
        super(OutMultiChoices, self).__init__()
        if activation == 'relu':
            self.activ = nn.ReLU()
        if activation == 'prelu':
            self.activ = nn.PReLU()
        if activation == 'elu':
            self.activ = nn.ELU()
        if activation == 'gelu':
            self.activ = nn.GELU()

        self.classifier = nn.Sequential(
                                        self.activ,
                                        nn.LayerNorm(embed_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(embed_dim, 1))


    def forward(self, x):
        out = self.classifier(x)
        return out
