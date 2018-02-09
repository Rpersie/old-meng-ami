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
from cnn_md import CNNAdversarialMultidecoder, CNNAdversarialVariationalMultidecoder
from activation_dict import ActivationDict
from utils.hao_data import HaoEvalDataset, write_kaldi_hao_ark, write_kaldi_hao_scp

run_start_t = time.clock()

# Parse command line args
run_mode = "ae"
adversarial = False

if len(sys.argv) == 3:
    run_mode = sys.argv[1]
    adversarial = True if sys.argv[2] == "true" else False
else:
    print("Usage: python cnn/scripts/activations_md.py <run mode> <adversarial true/false>", flush=True)
    sys.exit(1)
print("Running activation logging with mode %s" % run_mode, flush=True)
if adversarial:
    print("Using adversarial loss", flush=True)

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
    
if adversarial:
    adv_fc_sizes = []
    for res_str in os.environ["ADV_FC_DELIM"].split("_"):
        if len(res_str) > 0:
            adv_fc_sizes.append(int(res_str))
    adv_activation = os.environ["ADV_ACTIVATION"]

on_gpu = torch.cuda.is_available()
log_interval = 10   # Log results once for this many batches during training

# Set up input files and output directory
activations_holdout_scps = dict()
for decoder_class in decoder_classes:
    activations_holdout_scp_name = os.path.join(os.environ["CURRENT_FEATS"], "%s-activations_holdout-norm.blogmel.scp" % decoder_class)
    activations_holdout_scps[decoder_class] = activations_holdout_scp_name

if adversarial:
    output_dir = os.path.join(os.environ["ACTIVATIONS_DIR"], "adversarial_fc_%s_act_%s_%s_ratio%s" % (os.environ["ADV_FC_DELIM"],
                                                                                             adv_activation,
                                                                                             run_mode,
                                                                                             str(noise_ratio)))
else:
    output_dir = os.path.join(os.environ["ACTIVATIONS_DIR"], "%s_ratio%s" % (run_mode, noise_ratio))

# Fix random seed for debugging
torch.manual_seed(1)
if on_gpu:
    torch.cuda.manual_seed(1)
random.seed(1)

# Construct autoencoder with our parameters
print("Constructing model...", flush=True)
if run_mode == "ae":
    if adversarial:
        model = CNNAdversarialMultidecoder(freq_dim=freq_dim,
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
                                use_batch_norm=use_batch_norm,
                                decoder_classes=decoder_classes,
                                weight_init=weight_init,
                                adv_fc_sizes=adv_fc_sizes,
                                adv_activation=adv_activation)
    else:
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
                                use_batch_norm=use_batch_norm,
                                decoder_classes=decoder_classes,
                                weight_init=weight_init)
elif run_mode == "vae":
    if adversarial:
        model = CNNVariationalAdversarialMultidecoder(freq_dim=freq_dim,
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
                                use_batch_norm=use_batch_norm,
                                decoder_classes=decoder_classes,
                                weight_init=weight_init,
                                adv_fc_sizes=adv_fc_sizes,
                                adv_activation=adv_activation)
    else:
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
                                use_batch_norm=use_batch_norm,
                                decoder_classes=decoder_classes,
                                weight_init=weight_init)
else:
    print("Unknown train mode %s" % run_mode, flush=True)
    sys.exit(1)

if on_gpu:
    model.cuda()
print("Done constructing model.", flush=True)
print(model, flush=True)

# Load checkpoint (potentially trained on GPU) into CPU memory (hence the map_location)
print("Loading checkpoint...")
model_dir = os.environ["MODEL_DIR"]
if adversarial:
    best_ckpt_path = os.path.join(model_dir, "best_cnn_adversarial_fc_%s_act_%s_%s_ratio%s_md.pth.tar" % (os.environ["ADV_FC_DELIM"],
                                                                                              adv_activation,
                                                                                              run_mode,
                                                                                              str(noise_ratio)))
else:
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

print("Setting up activations holdout datasets...", flush=True)
activations_holdout_datasets = dict()
activations_holdout_loaders = dict()
for decoder_class in decoder_classes:
    current_dataset = HaoEvalDataset(activations_holdout_scps[decoder_class])
    activations_holdout_datasets[decoder_class] = current_dataset
    activations_holdout_loaders[decoder_class] = DataLoader(current_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            **loader_kwargs)
    print("Using %d activations holdout features (%d batches) for class %s" % (len(current_dataset),
                                                               len(activations_holdout_loaders[decoder_class]),
                                                               decoder_class),
          flush=True)

print("Done setting up data.", flush=True)

setup_end_t = time.clock()
print("Completed setup in %.3f seconds" % (setup_end_t - run_start_t), flush=True)

# Process activations holdout dataset
model.eval()
print("=> Processing activations holdout data...", flush=True)
for decoder_class in decoder_classes:
    process_start_t = time.clock()

    print("===> Processing class %s..." % decoder_class, flush=True)
    batches_processed = 0
    total_batches = len(activations_holdout_loaders[decoder_class])
        
    activations_dict = ActivationDict(layer_names=model.encoder_conv_activation_layers(),
                                      unit_counts=enc_channel_sizes,
                                      top_count=top_count)

    for batch_idx, (feats, targets, utt_ids) in enumerate(activations_holdout_loaders[decoder_class]):
        utt_id = utt_ids[0]     # Batch size 1; only one utterance

        # Run batch through target decoder
        feats_numpy = feats.numpy().reshape((-1, freq_dim))
        num_frames = feats_numpy.shape[0]
        decoded_feats = np.empty((num_frames, freq_dim))
        for i in range(num_frames):
            frame_spliced = np.zeros((time_dim, freq_dim))
            frame_spliced[left_context - min(i, left_context):left_context, :] = feats_numpy[i - min(i, left_context):i, :]
            frame_spliced[left_context, :] = feats_numpy[i, :]
            frame_spliced[left_context + 1:left_context + 1 + min(num_frames - i - 1, right_context), :] = feats_numpy[i + 1:i + 1 + min(num_frames - i - 1, right_context), :]
            frame_tensor = Variable(torch.FloatTensor(frame_spliced).view((1, 1, time_dim, freq_dim)))
            if on_gpu:
                frame_tensor = frame_tensor.cuda()

            layer_activations = model.get_encoder_conv_activations(frame_tensor)
            for layer_key in layer_activations:
                layer_activation_numpy = layer_activations[layer_key].cpu().data.numpy()
                for unit_idx in range(activations_dict.unit_counts[layer_key]):
                    avg_activation = np.average(layer_activation_numpy[0, unit_idx, :, :].reshape((-1)))
                    name = "%s_frame%d_%s" % (decoder_class, i, utt_id)
                    activations_dict.update_top(layer_key, unit_idx, avg_activation, name, frame_spliced)

        batches_processed += 1
        if batches_processed % log_interval == 0:
            print("=====> Logged activations for %d/%d batches (%.1f%%)]" % (batches_processed,
                                                                             total_batches,
                                                                             100.0 * batches_processed / total_batches),
                  flush=True)

    # Write average of features with top activations to output files
    print("Writing average of top activations to file...", flush=True)
    for layer_key in activations_dict.layer_names:
        with open(os.path.join(output_dir, "activations_holdout-%s-%s.ark" % (decoder_class, layer_key)), 'w') as ark_fd:
            for unit_idx in range(activations_dict.unit_counts[layer_key]):
                avg_feats = activations_dict.avg_feats(layer_key, unit_idx)
                write_kaldi_hao_ark(ark_fd, str(unit_idx), avg_feats)
        with open(os.path.join(output_dir, "activations_holdout-%s-%s.scp" % (decoder_class, layer_key)), 'w') as scp_fd:
            write_kaldi_hao_scp(scp_fd, os.path.join(output_dir, "activations_holdout-%s-%s.ark" % (decoder_class, layer_key)))

    process_end_t = time.clock()
    print("===> Processed class %s in %.3f seconds" % (decoder_class, process_end_t - process_start_t), flush=True)

print("=> Done with activations holdout data", flush=True)

run_end_t = time.clock()
print("Completed activations logging run in %.3f seconds" % (run_end_t - run_start_t), flush=True)
