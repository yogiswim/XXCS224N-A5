#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 
dropout_rate = 0.3
char_embedding_size = 50
max_word_length = 21

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.char_embedding_layer = nn.Embedding(len(vocab.char2id), char_embedding_size, vocab.char2id['<pad>'])
        self.cnn_layer = CNN(char_embedding_size, max_word_length, embed_size)
        self.highway = Highway(embed_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.embed_size = embed_size
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code
        
        ### YOUR CODE HERE for part 1f
        x_word_embeddings = []
        for word in input:
            x_char_embedding = self.char_embedding_layer(word)  # (batch_size, max_word_length, embed_size)
            x_reshaped = x_char_embedding.permute(0, 2, 1)
            x_conv_out = self.cnn_layer(x_reshaped)
            x_highway = self.highway(x_conv_out)
            x_word_embed = self.dropout(x_highway)
            x_word_embeddings.append(x_word_embed)

        x_word_embeddings = torch.stack(x_word_embeddings)
        return x_word_embeddings
        ### END YOUR CODE
