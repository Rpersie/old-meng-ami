from collections import OrderedDict
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Base class -- do not use directly!
class Multidecoder(nn.Module):
    def model_parameters(self, decoder_class):
        # Get parameters for just a specific decoder and the encoder
        if decoder_class not in self.decoder_classes:
            print("Decoder class \"%s\" not found in decoders: %s" % (decoder_class, self.decoder_classes),
                  flush=True)
            sys.exit(1)
        decoder_parameters = self.decoders[decoder_class].parameters()
        for param in decoder_parameters:
            yield param
        
        encoder_parameters = self.encoder.parameters()
        for param in encoder_parameters:
            yield param

    def encode(self, feats):
        return self.encoder(feats)

    def decode(self, z, decoder_class):
        return self.decoders[decoder_class](z)

    def forward_decoder(self, feats, decoder_class):
        latent = self.encode(feats.view(-1, self.input_dim))
        return self.decode(latent, decoder_class)



# CNN BASED MULTIDECODERS



# Simpler multidecoder design with 1D convolutional encoder/decoder layers
class CNN1DMultidecoder(Multidecoder):
    def __init__(self, feat_dim=80,
                       splicing=[5,5],
                       enc_channel_sizes=[],
                       enc_kernel_sizes=[],
                       enc_pool_sizes=[],
                       latent_dim=512,
                       dec_channel_sizes=[],
                       dec_kernel_sizes=[],
                       dec_pool_sizes=[],
                       activation="SELU",
                       decoder_classes=[""]):
        super(CNN1DMultidecoder, self).__init__()

        # Store initial parameters
        self.feat_dim = feat_dim
        self.splicing = splicing    # [left, right]
        self.enc_channel_sizes = enc_channel_sizes
        self.enc_kernel_sizes = enc_kernel_sizes
        self.enc_pool_sizes = enc_pool_sizes
        self.latent_dim = latent_dim
        self.dec_channel_sizes = dec_channel_sizes
        self.dec_kernel_sizes = dec_kernel_sizes
        self.dec_pool_sizes = dec_pool_sizes
        self.activation = getattr(nn, activation)()
        self.decoder_classes = decoder_classes

        # Construct encoder
        self.input_dim = (feat_dim * (sum(splicing) + 1))
        current_channels = 1
        current_length = self.input_dim
        self.encoder_layers = OrderedDict()
        for idx in range(len(enc_channel_sizes)):
            enc_channel_size = enc_channel_sizes[idx]
            enc_kernel_size = enc_kernel_sizes[idx]
            enc_pool_size = enc_pool_sizes[idx]

            self.encoder_layers["conv1d_%d" % idx] = nn.Conv1d(current_channels, enc_channel_size, enc_kernel_size)
            current_channels = enc_channel_size

            # Formula for length from http://pytorch.org/docs/master/nn.html#conv1d
            # Assumes stride = 1, padding = 0, dilation = 1
            current_length = (current_length - (enc_kernel_size - 1) - 1) + 1

            self.encoder_layers["bn_%d" % idx] = nn.BatchNorm1d(enc_channel_size)

            if enc_pool_size > 0:
                self.encoder_layers["maxpool1d_%d" % idx] = nn.MaxPool1d(enc_pool_size)

                # Formula for length from http://pytorch.org/docs/master/nn.html#maxpool1d
                # Assumes stride = enc_pool_size, padding = 0, dilation = 1
                current_length = int((current_length - (enc_pool_size - 1) - 1) / enc_pool_size) + 1

            self.encoder_layers["%s_%d" % (activation, idx)] = self.activation

        self.encoder_layers["lin_final"] = nn.Linear(current_channels * current_length, self.latent_dim)
        self.encoder = nn.Sequential(self.encoder_layers)

        # Construct decoders
        self.decoders = dict()
        self.decoder_layers = dict()
        for decoder_class in self.decoder_classes:
            current_channels = 1
            current_length = self.latent_dim
            self.decoder_layers[decoder_class] = OrderedDict()
            for idx in range(len(dec_channel_sizes)):
                dec_channel_size = dec_channel_sizes[idx]
                dec_kernel_size = dec_kernel_sizes[idx]
                dec_pool_size = dec_pool_sizes[idx]

                self.decoder_layers[decoder_class]["conv1d_%d" % idx] = nn.Conv1d(current_channels, dec_channel_size, dec_kernel_size)
                current_channels = dec_channel_size

                # Formula for length from http://pytorch.org/docs/master/nn.html#conv1d
                # Assumes stride = 1, padding = 0, dilation = 1
                current_length = (current_length - (dec_kernel_size - 1) - 1) + 1

                self.decoder_layers[decoder_class]["bn_%d" % idx] = nn.BatchNorm1d(dec_channel_size)

                if dec_pool_size > 0:
                    self.decoder_layers[decoder_class]["maxpool1d_%d" % idx] = nn.MaxPool1d(dec_pool_size)

                    # Formula for length from http://pytorch.org/docs/master/nn.html#maxpool1d
                    # Assumes stride = dec_pool_size, padding = 0, dilation = 1
                    current_length = int((current_length - (dec_pool_size - 1) - 1) / dec_pool_size) + 1
                
                self.decoder_layers[decoder_class]["%s_%d" % (activation, idx)] = self.activation

            self.decoder_layers[decoder_class]["lin_final"] = nn.Linear(current_channels * current_length, self.input_dim)
            self.decoders[decoder_class] = nn.Sequential(self.decoder_layers[decoder_class])

            self.add_module("decoder_%s" % decoder_class, self.decoders[decoder_class])
    
    # Overwritten to handle converting conv outputs to format for linear layers
    def encode(self, feats):
        channeled_feats = feats.view(-1, 1, self.input_dim)
        output = channeled_feats
        for i, (encoder_layer_name, encoder_layer) in enumerate(self.encoder_layers.items()):
            if encoder_layer_name == "lin_final":
                # Skip last (linear) layer
                break
            output = encoder_layer(output)
        converted_output = output.view(output.size()[0], -1)
        
        out = self.encoder_layers["lin_final"](converted_output)
        return out

    # Overwritten to handle converting conv outputs to format for linear layers
    def decode(self, z, decoder_class):
        channeled_z = z.view(-1, 1, self.latent_dim)
        output = channeled_z
        for i, (decoder_layer_name, decoder_layer) in enumerate(self.decoder_layers[decoder_class].items()):
            if decoder_layer_name == "lin_final":
                # Skip last (linear) layer
                break
            output = decoder_layer(output)
        converted_output = output.view(output.size()[0], -1)

        out = self.decoder_layers[decoder_class]["lin_final"](converted_output)
        return out
