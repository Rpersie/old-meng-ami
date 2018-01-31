import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

sys.path.append("./")
sys.path.append("./cnn")
from cnn_md import CNNMultidecoder, CNNVariationalMultidecoder
from activation_dict import ActivationDict
from utils.hao_data import HaoEvalDataset, write_kaldi_hao_ark, write_kaldi_hao_scp

run_start_t = time.clock()

# Parse command line args
run_mode = "ae"
if len(sys.argv) == 2:
    run_mode = sys.argv[1]
print("Running activation logging with mode %s" % run_mode, flush=True)

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
log_interval = 100   # Log results once for this many batches during training

# Set up input files and output directory
dev_scps = dict()
for decoder_class in decoder_classes:
    dev_scp_name = os.path.join(os.environ["CURRENT_FEATS"], "%s-dev-norm.blogmel.scp" % decoder_class)
    dev_scps[decoder_class] = dev_scp_name

if run_mode in ["dae", "dvae"]:
    output_dir = os.path.join(os.environ["ACTIVATIONS_DIR"], "%s_ratio%s" % (run_mode, noise_ratio))
else:
    output_dir = os.path.join(os.environ["ACTIVATIONS_DIR"], run_mode)

# Fix random seed for debugging
torch.manual_seed(1)
if on_gpu:
    torch.cuda.manual_seed(1)
random.seed(1)

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
    print("Unknown activation logging mode %s" % run_mode, flush=True)
    sys.exit(1)

if on_gpu:
    model.cuda()
print("Done constructing model.", flush=True)
print(model, flush=True)

# Load checkpoint (potentially trained on GPU) into CPU memory (hence the map_location)
print("Loading checkpoint...")
model_dir = os.environ["MODEL_DIR"]
best_ckpt_path = os.path.join(model_dir, "best_cnn_%s_ratio%s_md.pth.tar" % (run_mode, str(noise_ratio)))
checkpoint = torch.load(best_ckpt_path, map_location=lambda storage,loc: storage)

# Set up model state and set to eval mode (i.e. disable batch norm)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print("Loaded checkpoint; best model ready now.")



# LOG ACTIVATIONS USING MULTIDECODER



top_count = int(os.environ["TOP_COUNT"])

print("Setting up data...", flush=True)
loader_kwargs = {"num_workers": 1, "pin_memory": True} if on_gpu else {}

print("Setting up dev datasets...", flush=True)
dev_datasets = dict()
dev_loaders = dict()
for decoder_class in decoder_classes:
    current_dataset = HaoEvalDataset(dev_scps[decoder_class])
    dev_datasets[decoder_class] = current_dataset
    dev_loaders[decoder_class] = DataLoader(current_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            **loader_kwargs)
    print("Using %d dev features (%d batches) for class %s" % (len(current_dataset),
                                                               len(dev_loaders[decoder_class]),
                                                               decoder_class),
          flush=True)

print("Done setting up data.", flush=True)

setup_end_t = time.clock()
print("Completed setup in %.3f seconds" % (setup_end_t - run_start_t), flush=True)

# Process dev dataset
model.eval()
print("=> Processing dev data...", flush=True)
for decoder_class in decoder_classes:
    process_start_t = time.clock()

    print("===> Processing class %s..." % decoder_class, flush=True)

    for batch_idx, (feats, targets, utt_ids) in enumerate(dev_loaders[decoder_class]):
        utt_id = utt_ids[0]     # Batch size 1; only one utterance

        activations_dict = ActivationDict(layer_names=model.encoder_conv_activation_layers(),
                                          top_count=top_count)

        # Run batch through target decoder
        feats_numpy = feats.numpy().reshape((-1, freq_dim))
        num_frames = feats_numpy.shape[0]
        decoded_feats = np.empty((num_frames, freq_dim))
        for i in range(num_frames):
            frame_spliced = np.zeros((time_dim, freq_dim))
            frame_spliced[left_context - min(i, left_context):left_context, :] = feats_numpy[i - min(i, left_context):i, :]
            frame_spliced[left_context, :] = feats_numpy[i, :]
            frame_spliced[left_context + 1:left_context + 1 + min(num_frames - i - 1, right_context), :] = feats_numpy[i + 1:i + 1 + min(num_frames - i - 1, right_context), :]
            frame_tensor = Variable(torch.FloatTensor(frame_spliced))
            if on_gpu:
                frame_tensor = frame_tensor.cuda()

            layer_activations = model.get_encoder_conv_activations(frame_tensor, target_class)
            for layer_key in layer_activations:
                layer_activation_numpy = layer_activations[layer_key].cpu().data.numpy().reshape((num_frames, -1)) 
                activations_dict.update_top(layer_key, layer_activation_numpy)

        batches_processed += 1
        if batches_processed % log_interval == 0:
            print("===> Logged activations for %d/%d batches (%.1f%%)]" % (batches_processed,
                                                              total_batches,
                                                              100.0 * batches_processed / total_batches),
                  flush=True)

    # Write features with top activations to output files
    activation_ark_fds = {open(os.path.join(output_dir, "dev-%s-%s.ark" % (decoder_class, layer_key)), 'w') for layer_key in model.encoder_conv_activation_layers()}
    for layer_key in activations_dict.layer_names:
        top_activations = activations_dict.top_ordered(layer_key)  
        ark_fd = activation_ark_fds[layer_key]
        for (activation_name, feats) in top_activations:
            write_kaldi_hao_ark(ark_fd, activations_name, feats)
        ark_fd.close()

    # Create corresponding SCP files
    print("===> Writing SCPs...", flush=True)
    activation_scp_fds = {open(os.path.join(output_dir, "dev-%s-%s.scp" % (decoder_class, layer_key)), 'w') for layer_key in model.encoder_conv_activation_layers()}
    for layer_key in model.encoder_conv_activation_layers():
        scp_fd = activation_scp_fds[layer_key]
        write_kaldi_hao_scp(scp_fd, os.path.join(output_dir, "dev-%s-%s.ark" % (decoder_class, layer_key)))
        scp_fd.close()

    process_end_t = time.clock()
    print("===> Processed class %s in %.3f seconds" % (decoder_class, process_end_t - process_start_t), flush=True)

print("=> Done with dev data", flush=True)

run_end_t = time.clock()
print("Completed activations logging run in %.3f seconds" % (run_end_t - run_start_t), flush=True)
