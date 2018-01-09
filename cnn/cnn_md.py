from collections import OrderedDict
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



# CNN BASED MULTIDECODERS



# Simpler multidecoder design with 1D convolutional encoder/decoder layers
class CNNMultidecoder(nn.Module):
    def __init__(self, freq_dim=80,
                       splicing=[5,5],
                       enc_channel_sizes=[],
                       enc_kernel_sizes=[],
                       enc_pool_sizes=[],
                       enc_fc_sizes=[],
                       latent_dim=512,
                       dec_fc_sizes=[],
                       dec_channel_sizes=[],
                       dec_kernel_sizes=[],
                       dec_pool_sizes=[],
                       activation="SELU",
                       decoder_classes=[""]):
        super(CNNMultidecoder, self).__init__()

        # Store initial parameters
        self.freq_dim = freq_dim
        self.splicing = splicing    # [left, right]
        self.time_dim = (sum(splicing) + 1)

        self.enc_channel_sizes = enc_channel_sizes
        self.enc_kernel_sizes = enc_kernel_sizes
        self.enc_pool_sizes = enc_pool_sizes
        self.enc_fc_sizes = enc_fc_sizes

        self.latent_dim = latent_dim

        self.dec_fc_sizes = dec_fc_sizes
        self.dec_channel_sizes = dec_channel_sizes
        self.dec_kernel_sizes = dec_kernel_sizes
        self.dec_pool_sizes = dec_pool_sizes

        self.activation = activation
        self.decoder_classes = decoder_classes


        # STEP 1: Construct encoder


        current_channels = 1
        current_height = self.time_dim
        current_width = self.freq_dim

        # Convolutional stage
        self.encoder_conv_layers = OrderedDict()
        for idx in range(len(enc_channel_sizes)):
            enc_channel_size = enc_channel_sizes[idx]
            enc_kernel_size = enc_kernel_sizes[idx]
            enc_pool_size = enc_pool_sizes[idx]

            self.encoder_conv_layers["conv2d_%d" % idx] = nn.Conv2d(current_channels,
                                                                    enc_channel_size,
                                                                    enc_kernel_size)
            current_channels = enc_channel_size

            # Formula from http://pytorch.org/docs/master/nn.html#conv2d
            # Assumes stride = 1, padding = 0, dilation = 1
            current_height = (current_height - (enc_kernel_size - 1) - 1) + 1
            current_width = (current_width - (enc_kernel_size - 1) - 1) + 1

            self.encoder_conv_layers["batchnorm2d_%d" % idx] = nn.BatchNorm2d(enc_channel_size)

            self.encoder_conv_layers["%s_%d" % (self.activation, idx)] = getattr(nn, self.activation)()
            
            if enc_pool_size > 0:
                # Pool only in frequency direction (i.e. kernel and stride 1 in time dimension)
                # Return indices as well (useful for unpooling: see
                #   http://pytorch.org/docs/master/nn.html#maxunpool2d)
                self.encoder_conv_layers["maxpool2d_%d" % idx] = nn.MaxPool2d((1, enc_pool_size),
                                                                              return_indices=True)
                
                # Formula from http://pytorch.org/docs/master/nn.html#maxpool2d 
                # Assumes stride = enc_pool_size (default), padding = 0, dilation = 1
                current_height = current_height     # No change in time dimension!
                current_width = int((current_width - (enc_pool_size - 1) - 1) / enc_pool_size) + 1

        self.encoder_conv = nn.Sequential(self.encoder_conv_layers)
        
        # Fully-connected stage
        self.encoder_fc_layers = OrderedDict()
        current_fc_dim = current_channels * current_height * current_width
        for idx in range(len(enc_fc_sizes)):
            enc_fc_size = enc_fc_sizes[idx]
            
            self.encoder_fc_layers["lin_%d" % idx] = nn.Linear(current_fc_dim, enc_fc_size)
            self.encoder_fc_layers["%s_%d" % (self.activation, idx)] = getattr(nn, self.activation)()
            
            current_fc_dim = enc_fc_size

        self.encoder_fc_layers["lin_final"] = nn.Linear(current_fc_dim, self.latent_dim)
        self.encoder_fc = nn.Sequential(self.encoder_fc_layers)


        # STEP 2: Construct decoders


        self.decoder_fc = dict()
        self.decoder_fc_layers = dict()
        self.decoder_deconv = dict()
        self.decoder_deconv_layers = dict()

        # Save values from encoder stage
        input_channels = current_channels
        input_height = current_height
        input_width = current_width

        for decoder_class in self.decoder_classes:
            # Fully-connected stage
            self.decoder_fc_layers[decoder_class] = OrderedDict()
            current_fc_dim = self.latent_dim
            for idx in range(len(dec_fc_sizes)):
                dec_fc_size = dec_fc_sizes[idx]

                self.decoder_fc_layers[decoder_class]["lin_%d" % idx] = nn.Linear(current_fc_dim, dec_fc_size)
                self.decoder_fc_layers[decoder_class]["%s_%d" % (self.activation, idx)] = getattr(nn, self.activation)()

                current_fc_dim = dec_fc_size
        
            self.decoder_fc_layers[decoder_class]["lin_final"] = nn.Linear(current_fc_dim,
                                                                           input_channels * input_height * input_width)
            self.decoder_fc[decoder_class] = nn.Sequential(self.decoder_fc_layers[decoder_class])
            self.add_module("decoder_fc_%s" % decoder_class, self.decoder_fc[decoder_class])

            # Deconvolution stage
            current_height = input_height
            current_width = input_width

            self.decoder_deconv_layers[decoder_class] = OrderedDict()
            for idx in range(len(dec_channel_sizes)):
                dec_channel_size = dec_channel_sizes[idx]
                dec_kernel_size = dec_kernel_sizes[idx]
                dec_pool_size = dec_pool_sizes[idx]
                
                if dec_pool_size > 0:
                    # Un-pool only in frequency direction (i.e. kernel and stride 1 in time dimension)
                    self.decoder_deconv_layers[decoder_class]["maxunpool2d_%d" % idx] = nn.MaxUnpool2d((1, dec_pool_size))
                    
                    # Formula from http://pytorch.org/docs/master/nn.html#maxunpool2d 
                    # Assumes stride = dec_pool_size (default), padding = 0, dilation = 1
                    current_height = current_height     # No change in time dimension!
                    current_width = current_width * dec_pool_size 

                # Re-pad signal to "de-convolve"
                # https://pgaleone.eu/neural-networks/2016/11/24/convolutional-autoencoders/
                padding = dec_kernel_size - 1 
                output_channels = 1 if idx == len(dec_channel_sizes) - 1 else dec_channel_sizes[idx + 1]
                self.decoder_deconv_layers[decoder_class]["conv2d_%d" % idx] = nn.Conv2d(dec_channel_size,
                                                                        output_channels,
                                                                        dec_kernel_size,
                                                                        padding=padding)

                # Formula for length from http://pytorch.org/docs/master/nn.html#conv2d
                # Assumes stride = 1, dilation = 1
                current_height = current_height + padding
                current_width = current_width + padding

                self.decoder_deconv_layers[decoder_class]["batchnorm2d_%d" % idx] = nn.BatchNorm2d(output_channels)
                self.decoder_deconv_layers[decoder_class]["%s_%d" % (self.activation, idx)] = getattr(nn, self.activation)()

            self.decoder_deconv[decoder_class] = nn.Sequential(self.decoder_deconv_layers[decoder_class])
            self.add_module("decoder_deconv_%s" % decoder_class, self.decoder_deconv[decoder_class])
    
    def model_parameters(self, decoder_class):
        # Get parameters for just a specific decoder and the encoder
        if decoder_class not in self.decoder_classes:
            print("Decoder class \"%s\" not found in decoders: %s" % (decoder_class, self.decoder_classes),
                  flush=True)
            sys.exit(1)
        decoder_deconv_parameters = self.decoder_deconv[decoder_class].parameters()
        for param in decoder_deconv_parameters:
            yield param
        decoder_fc_parameters = self.decoder_fc[decoder_class].parameters()
        for param in decoder_fc_parameters:
            yield param
        
        encoder_fc_parameters = self.encoder_fc.parameters()
        for param in encoder_fc_parameters:
            yield param
        encoder_conv_parameters = self.encoder_conv.parameters()
        for param in encoder_conv_parameters:
            yield param
    
    def encode(self, feats):
        # Need to go layer-by-layer to get pooling indices
        print("ENCODING", flush=True)
        pooling_indices = []    
        unpool_sizes = []
        conv_encoded = feats
        print(conv_encoded.size(), flush=True)
        for i, (encoder_conv_layer_name, encoder_conv_layer) in enumerate(self.encoder_conv_layers.items()):
            if "maxpool2d" in encoder_conv_layer_name:
                unpool_sizes.append(conv_encoded.size())
                print("Using unpool size %s" % str(conv_encoded.size()), flush=True)
                conv_encoded, new_pooling_indices = encoder_conv_layer(conv_encoded)
                print("Using indices %s" % str(new_pooling_indices.size()), flush=True)
                pooling_indices.append(new_pooling_indices)
            else:
                conv_encoded = encoder_conv_layer(conv_encoded)
            print(conv_encoded.size(), flush=True)
        fc_input_size = conv_encoded.size()
        conv_encoded_vec = conv_encoded.view(conv_encoded.size()[0], -1)
        return (self.encoder_fc(conv_encoded_vec), fc_input_size, unpool_sizes, pooling_indices)

    def decode(self, z, decoder_class, fc_input_size, unpool_sizes, pooling_indices):
        print("DECODING", flush=True)
        fc_decoded = self.decoder_fc[decoder_class](z)
        print(fc_decoded.size(), flush=True)
        fc_decoded_mat = fc_decoded.view(fc_input_size) 
        print(fc_decoded_mat.size(), flush=True)

        # Need to go layer-by-layer to insert pooling indices into unpooling layers
        output = fc_decoded_mat
        for i, (decoder_deconv_layer_name, decoder_deconv_layer) in enumerate(self.decoder_deconv_layers[decoder_class].items()):
            if "maxunpool2d" in decoder_deconv_layer_name:
                current_pooling_indices = pooling_indices.pop()
                current_unpool_size = unpool_sizes.pop()
                print("Using unpool size %s" % str(current_unpool_size), flush=True)
                print("Using indices %s" % str(current_pooling_indices.size()), flush=True)
                output = decoder_deconv_layer(output,
                                              current_pooling_indices,
                                              output_size=current_unpool_size)
            else:
                output = decoder_deconv_layer(output)
            print(output.size(), flush=True)

        return output
    
    def forward_decoder(self, feats, decoder_class):
        latent, fc_input_size, unpool_sizes, pooling_indices = self.encode(feats.view(-1,
                                                                           1,
                                                                           self.time_dim,
                                                                           self.freq_dim))
        return self.decode(latent, decoder_class, fc_input_size, unpool_sizes, pooling_indices)
