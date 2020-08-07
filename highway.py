#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d

import torch.nn as nn
import torch.functional as F
import torch

class Highway(nn.Module):

    def __init__(self, embedding_size):
        """
        Init Highway model
        :param embedding_size (int): Embedding size (dimensionality)
        """
        super(Highway, self).__init__()
        self.projection = nn.Linear(embedding_size, embedding_size)
        self.gate = nn.Linear(embedding_size, embedding_size)

    def forward(self, x_conv_out):
        """
        Map from x_conv_out to x_highway
        :param x_conv_out: Tensor output from cnn layer. Input size (batch_size, embedding_size)
        :return: x_highway: Tensor output from Highway network. Output size (batch_size, embedding_size)
        """
        x_proj = torch.nn.functional.relu(self.projection(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))

        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        return x_highway

### END YOUR CODE 

