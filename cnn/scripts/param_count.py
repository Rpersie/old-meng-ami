import os
import sys

import numpy as np
import torch

sys.path.append("./")
sys.path.append("./cnn")
from cnn_md import CNNMultidecoder, CNNVariationalMultidecoder

# Parse command line args
run_mode = "ae"
if len(sys.argv) == 2:
    run_mode = sys.argv[1]
print("Running augmentation with mode %s" % run_mode, flush=True)

# Set up noising
noise_ratio = float(os.environ["NOISE_RATIO"])
print("Noise ratio: %.3f%% of input features" % (noise_ratio * 100.0), flush=True)

# Uses some structure from https://github.com/pytorch/examples/blob/master/vae/main.py

# Set up features
feature_name = "fbank"
feat_dim = int(os.environ["FEAT_DIM"])
left_context = int(os.environ["LEFT_CONTEXT"])
right_context = int(os.environ["RIGHT_CONTEXT"])

freq_dim = feat_dim
time_dim = (left_context + right_context + 1)

# Read in parameters to use for our network
enc_channel_sizes = []
for res_str in os.environ["ENC_CHANNELS_DELIM"].split("_"):
    if len(res_str) > 0:
        enc_channel_sizes.append(int(res_str))
enc_kernel_sizes = []
for res_str in os.environ["ENC_KERNELS_DELIM"].split("_"):
    if len(res_str) > 0:
        enc_kernel_sizes.append(int(res_str))
enc_pool_sizes = []
for res_str in os.environ["ENC_POOLS_DELIM"].split("_"):
    if len(res_str) > 0:
        enc_pool_sizes.append(int(res_str))
enc_fc_sizes = []
for res_str in os.environ["ENC_FC_DELIM"].split("_"):
    if len(res_str) > 0:
        enc_fc_sizes.append(int(res_str))

latent_dim = int(os.environ["LATENT_DIM"])

dec_fc_sizes = []
for res_str in os.environ["DEC_FC_DELIM"].split("_"):
    if len(res_str) > 0:
        dec_fc_sizes.append(int(res_str))
dec_channel_sizes = []
for res_str in os.environ["DEC_CHANNELS_DELIM"].split("_"):
    if len(res_str) > 0:
        dec_channel_sizes.append(int(res_str))
dec_kernel_sizes = []
for res_str in os.environ["DEC_KERNELS_DELIM"].split("_"):
    if len(res_str) > 0:
        dec_kernel_sizes.append(int(res_str))
dec_pool_sizes = []
for res_str in os.environ["DEC_POOLS_DELIM"].split("_"):
    if len(res_str) > 0:
        dec_pool_sizes.append(int(res_str))

activation = os.environ["ACTIVATION_FUNC"]
decoder_classes = []
for res_str in os.environ["DECODER_CLASSES_DELIM"].split("_"):
    if len(res_str) > 0:
        decoder_classes.append(res_str)
use_batch_norm = True if os.environ["USE_BATCH_NORM"] == "true" else False
weight_init = os.environ["WEIGHT_INIT"]

on_gpu = torch.cuda.is_available()

# Fix random seed for debugging
torch.manual_seed(1)
if on_gpu:
    torch.cuda.manual_seed(1)

# Construct autoencoder with our parameters
print("Constructing model...", flush=True)
if run_mode == "ae":
    model = CNNMultidecoder(freq_dim=freq_dim,
                            splicing=[left_context, right_context], 
                            enc_channel_sizes=enc_channel_sizes,
                            enc_kernel_sizes=enc_kernel_sizes,
                            enc_pool_sizes=enc_pool_sizes,
                            enc_fc_sizes=enc_fc_sizes,
                            latent_dim=latent_dim,
                            dec_fc_sizes=dec_fc_sizes,
                            dec_channel_sizes=dec_channel_sizes,
                            dec_kernel_sizes=dec_kernel_sizes,
                            dec_pool_sizes=dec_pool_sizes,
                            activation=activation,
                            decoder_classes=decoder_classes,
                            use_batch_norm=use_batch_norm,
                            weight_init=weight_init)
elif run_mode == "vae":
    model = CNNVariationalMultidecoder(freq_dim=freq_dim,
                            splicing=[left_context, right_context], 
                            enc_channel_sizes=enc_channel_sizes,
                            enc_kernel_sizes=enc_kernel_sizes,
                            enc_pool_sizes=enc_pool_sizes,
                            enc_fc_sizes=enc_fc_sizes,
                            latent_dim=latent_dim,
                            dec_fc_sizes=dec_fc_sizes,
                            dec_channel_sizes=dec_channel_sizes,
                            dec_kernel_sizes=dec_kernel_sizes,
                            dec_pool_sizes=dec_pool_sizes,
                            activation=activation,
                            decoder_classes=decoder_classes,
                            use_batch_norm=use_batch_norm,
                            weight_init=weight_init)
else:
    print("Unknown augment mode %s" % run_mode, flush=True)
    sys.exit(1)

if on_gpu:
    model.cuda()
print("Done constructing model.", flush=True)
print(model, flush=True)

# Count number of trainable parameters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Model has %d trainable parameters" % params, flush=True)
