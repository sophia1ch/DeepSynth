import logging 
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence

"""
IO = [[I1, ...,Ik], O]
I1, ..., Ik, O are lists
IOs = [IO1, IO2, ..., IOn]
task = (IOs1, program1)
tasks = [task1, task2, ..., taskp]
"""
class SimpleEmbedding(nn.Module):
    def __init__(self,
                 IOEncoder,
                 output_dimension,
                 size_hidden,
                 ):
        super(SimpleEmbedding, self).__init__()

        self.IOEncoder = IOEncoder
        self.lexicon_size = IOEncoder.lexicon_size
        self.output_dimension = output_dimension

        embedding = nn.Embedding(self.lexicon_size, size_hidden)
        self.embedding = embedding

        self.hidden = nn.Sequential(
            nn.Linear(size_hidden, size_hidden),
            nn.LeakyReLU(),
            nn.Linear(size_hidden, output_dimension),
            nn.LeakyReLU(),
        )

    def forward_IOs(self, IOs):
        """
        returns a tensor of shape 
        (len(IOs), self.IOEncoder.output_dimension, self.output_dimension)
        """
        e = self.IOEncoder.encode_IOs(IOs)
        logging.debug("encoding size: {}".format(e.size()))
        e = self.embedding(e)
        logging.debug("embedding size: {}".format(e.size()))
        e = self.hidden(e)
        e = torch.mean(e, 0)
        assert(e.size() == (self.IOEncoder.output_dimension, self.output_dimension))
        return torch.flatten(e)

    def forward(self, batch_IOs):
        """
        returns a tensor of shape 
        (len(batch_IOs), self.IOEncoder.output_dimension, self.output_dimension)
        """
        res = torch.stack([self.forward_IOs(IOs) for IOs in batch_IOs])
        assert(res.size() == (len(batch_IOs), self.IOEncoder.output_dimension * self.output_dimension))
        return res

class RNNEmbedding(nn.Module):
    def __init__(self,
                 IOEncoder,
                 output_dimension,
                 size_hidden,
                 number_layers_RNN,
                 ):
        super(RNNEmbedding, self).__init__()

        self.IOEncoder = IOEncoder
        self.lexicon_size = IOEncoder.lexicon_size
        self.output_dimension = output_dimension
        self.size_hidden = size_hidden

        embedding = nn.Embedding(self.lexicon_size, size_hidden)
        self.embedding = embedding

        Hin = size_hidden * IOEncoder.output_dimension
        Hout = IOEncoder.output_dimension * output_dimension
        self.RNN_layer = nn.GRU(Hin, 
            Hout,
            number_layers_RNN, 
            batch_first = True,
        )

    def _forward_IOs(self, IOs):
        """
        returns a tensor of shape 
        (len(IOs), self.IOEncoder.output_dimension, self.output_dimension)
        """        
        e = self.IOEncoder.encode_IOs(IOs)
        logging.debug("encoding size: {}".format(e.size()))
        e = self.embedding(e)
        logging.debug("embedding size: {}".format(e.size()))
        assert e.size() == (len(IOs), self.IOEncoder.output_dimension, self.size_hidden),\
         "size not equal to: {} {} {}".format(len(IOs), self.IOEncoder.output_dimension, self.size_hidden)
        e = torch.flatten(e, start_dim = 1)
        e = torch.unsqueeze(e, 0)
        e,_ = self.RNN_layer(e)
        e = torch.squeeze(torch.squeeze(e, 0)[-1,:],0)
        assert e.size() == (self.IOEncoder.output_dimension * self.output_dimension, ),\
         "size not equal to: {}".format(self.IOEncoder.output_dimension * self.output_dimension)
        return e

    def forward(self, batch_IOs):
        """
        returns a tensor of shape 
        (len(batch_IOs), self.IOEncoder.output_dimension, self.output_dimension)
        """
        res = torch.stack([self._forward_IOs(IOs) for IOs in batch_IOs])
        assert(res.size() == (len(batch_IOs), self.IOEncoder.output_dimension * self.output_dimension))
        return res
   


class ZendoRNNEmbedding(nn.Module):
    def __init__(self,
                 IOEncoder,
                 output_dimension,
                 size_hidden,
                 number_layers_RNN):
        super().__init__()

        self.IOEncoder = IOEncoder
        self.lexicon_size = IOEncoder.lexicon_size
        self.output_dimension = output_dimension
        self.size_hidden = size_hidden

        print("✅ Symbol count for embedding:", IOEncoder.lexicon_size)
        self.embedding = nn.Embedding(self.lexicon_size, size_hidden)

        # One sequence = all tokens in one IO (excluding label)
        sequence_length = IOEncoder.output_dimension  # = max_objects * vector_length
        self.sequence_length = sequence_length

        self.rnn = nn.GRU(
            input_size=size_hidden,
            hidden_size=output_dimension,
            num_layers=number_layers_RNN,
            batch_first=True
        )

    def _forward_IOs(self, IOs):
        """
        IOs = [[structure_dict, label], ...]
        returns a flat tensor of shape (output_dimension,)
        """
        encoded = self.IOEncoder.encode_IOs(IOs)  # (num_IOs, vector_length + 1)
        logging.debug(f"Encoding shape: {encoded.shape}")

        # Remove label column
        encoded = encoded[:, :-1]  # shape: (num_IOs, input_dim)
        
        embedded = self.embedding(encoded)  # (num_IOs, input_dim, size_hidden)

        # Flatten input: treat each IO as a "sequence" of embedded tokens
        # RNN expects shape (batch, seq_len, embed_size), so:
        input_to_rnn = embedded  # already shape (num_IOs, input_dim, size_hidden)

        output, hidden = self.rnn(input_to_rnn)  # output: (num_IOs, input_dim, rnn_hidden)
        
        # Use the final RNN hidden state from the last layer
        final_hidden = hidden[-1]  # shape: (num_IOs, rnn_hidden)

        # Flatten by averaging across IOs
        return final_hidden.mean(dim=0)  # shape: (rnn_hidden,)


    def forward(self, batch_IOs):
        """
        Returns shape: (batch_size, output_dimension)
        """
        return torch.stack([self._forward_IOs(IOs) for IOs in batch_IOs])
