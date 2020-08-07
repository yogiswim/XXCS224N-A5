#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size)

        vocab_size = len(target_vocab.char2id)
        self.char_output_projection = nn.Linear(hidden_size, vocab_size)
        self.padding_idx = target_vocab.char2id['<pad>']
        self.decoderCharEmb = nn.Embedding(vocab_size, char_embedding_size, self.padding_idx)
        self.target_vocab = target_vocab
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        char_embedding = self.decoderCharEmb(input)
        h_t, dec_hidden = self.charDecoder(char_embedding, dec_hidden)
        s_t = self.char_output_projection(h_t)

        return s_t, dec_hidden
        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        s_t, dec_hidden = self.forward(char_sequence[:-1], dec_hidden)
        cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx, reduction="sum")

        loss = cross_entropy_loss(s_t.permute(1, 2, 0), char_sequence[1:].permute(1, 0))
        return loss

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        output_words = []

        batch_size = initialStates[0].shape[1]
        start_word_idx = self.target_vocab.start_of_word
        current_char = [[start_word_idx] * batch_size]

        dec_hidden = initialStates
        current_char_tensor = torch.tensor(current_char, device=device)
        for t in range(max_length):
            s_t, dec_hidden = self.forward(current_char_tensor, dec_hidden) # st (1, batch, self.vocab_size)
            current_char_tensor = s_t.argmax(-1) # (1, batch)
            current_char_list = current_char_tensor.squeeze(0).cpu().tolist()
            output_words.append(current_char_list) # List[(1, batch)]

        result = []
        for i in range(batch_size):
            word = ''
            for t in range(max_length):
                char_id = output_words[t][i]
                if char_id != self.target_vocab.end_of_word:
                    letter = self.target_vocab.id2char[char_id]
                    word += letter
                else:
                    break
            result.append(word)

        return result

        
        ### END YOUR CODE

