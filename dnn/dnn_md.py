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



# DNN BASED MULTIDECODERS



# Simpler multidecoder design with fully-connected encoder/decoder layers
class DNNMultidecoder(Multidecoder):
    def __init__(self, feat_dim=80,
                       splicing=[5,5],
                       enc_layer_sizes=[],
                       latent_dim=64,
                       dec_layer_sizes=[],
                       activation="PReLU",
                       dropout=0.0,
                       decoder_classes=[""]):
        super(DNNMultidecoder, self).__init__()

        # Store initial parameters
        self.feat_dim = feat_dim
        self.splicing = splicing    # [left, right]
        self.enc_layer_sizes = enc_layer_sizes
        self.latent_dim = latent_dim
        self.dec_layer_sizes = dec_layer_sizes
        self.activation = getattr(nn, activation)()
        self.dropout = dropout
        self.decoder_classes = decoder_classes

        # Construct encoder
        self.input_dim = (feat_dim * (sum(splicing) + 1))
        current_dim = self.input_dim
        self.encoder_layers = OrderedDict()
        for idx in range(len(enc_layer_sizes)):
            enc_layer_size = enc_layer_sizes[idx]
            self.encoder_layers["lin_%d" % idx] = nn.Linear(current_dim, enc_layer_size)
            self.encoder_layers["%s_%d" % (activation, idx)] = self.activation
            self.encoder_layers["dropout_%d" % idx] = nn.Dropout(p=self.dropout)
            current_dim = enc_layer_size
        self.encoder_layers["lin_final"] = nn.Linear(current_dim, self.latent_dim)
        self.encoder = nn.Sequential(self.encoder_layers)

        # Construct decoders
        self.decoders = dict()
        self.decoder_layers = dict()
        for decoder_class in self.decoder_classes:
            current_dim = self.latent_dim
            self.decoder_layers[decoder_class] = OrderedDict()
            for idx in range(len(dec_layer_sizes)):
                dec_layer_size = dec_layer_sizes[idx]
                self.decoder_layers[decoder_class]["lin_%d" % idx] = nn.Linear(current_dim, dec_layer_size)
                self.decoder_layers[decoder_class]["%s_%d" % (activation, idx)] = self.activation
                self.decoder_layers[decoder_class]["dropout_%d" % idx] = nn.Dropout(p=self.dropout)
                current_dim = dec_layer_size
            self.decoder_layers[decoder_class]["lin_final"] = nn.Linear(current_dim, self.input_dim)
            self.decoders[decoder_class] = nn.Sequential(self.decoder_layers[decoder_class])
            self.add_module("decoder_%s" % decoder_class, self.decoders[decoder_class])
