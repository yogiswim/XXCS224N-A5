#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

kernel_size = 5

class CNN(nn.Module):
    def __init__(self, char_embedding, m_word, f):
        """
        Init of the CNN model
        :param char_embedding: char embedding size (dimensionality)
        :param m_word: max word length of sentence
        :param f: filter dimensionality
        """
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(char_embedding, f, kernel_size)
        self.maxpool = nn.MaxPool1d(m_word - kernel_size + 1)

    def forward(self, x_reshaped):
        """
        Perform convolution operation on embedding
        :param x_reshaped: tensor with dimension (batch_size, char_embedding, m_word)
        :return: x_conv_out: tensor with dimension (batch_size, word_embedding)
        """
        x_conv = self.conv1d(x_reshaped)
        x_conv_out = torch.squeeze(self.maxpool(F.relu(x_conv)), -1)
        return x_conv_out



### END YOUR CODE

