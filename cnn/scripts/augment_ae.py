import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

sys.path.append("./")
sys.path.append("./cnn")
from cnn_md import CNNMultidecoder
from utils.hao_data import HaoEvalDataset, write_kaldi_hao_ark, write_kaldi_hao_scp

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

on_gpu = torch.cuda.is_available()
log_interval = 100   # Log results once for this many batches during training

# Set up input files and output directory
training_scps = dict()
for decoder_class in decoder_classes:
    training_scp_name = os.path.join(os.environ["CURRENT_FEATS"], "%s-train-norm.blogmel.scp" % decoder_class)
    training_scps[decoder_class] = training_scp_name

dev_scps = dict()
for decoder_class in decoder_classes:
    dev_scp_name = os.path.join(os.environ["CURRENT_FEATS"], "%s-dev-norm.blogmel.scp" % decoder_class)
    dev_scps[decoder_class] = dev_scp_name

output_dir = os.environ["AUGMENTED_DATA_DIR"]

# Fix random seed for debugging
torch.manual_seed(1)
if on_gpu:
    torch.cuda.manual_seed(1)
random.seed(1)

# Construct autoencoder with our parameters
print("Constructing model...", flush=True)
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
                        decoder_classes=decoder_classes)
if on_gpu:
    model.cuda()
print("Done constructing model.", flush=True)
print(model, flush=True)

# Load checkpoint (potentially trained on GPU) into CPU memory (hence the map_location)
print("Loading checkpoint...")
checkpoint_path = os.path.join(os.environ["MODEL_DIR"], "best_cnn_ae_md.pth.tar")
checkpoint = torch.load(checkpoint_path, map_location=lambda storage,loc: storage)

# Set up model state and set to eval mode (i.e. disable batch norm)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print("Loaded checkpoint; best model ready now.")



# PERFORM DATA AUGMENTATION USING MULTIDECODER



print("Setting up data...", flush=True)
loader_kwargs = {"num_workers": 1, "pin_memory": True} if on_gpu else {}

print("Setting up training datasets...", flush=True)
training_datasets = dict()
training_loaders = dict()
for decoder_class in decoder_classes:
    current_dataset = HaoEvalDataset(training_scps[decoder_class])
    training_datasets[decoder_class] = current_dataset
    training_loaders[decoder_class] = DataLoader(current_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 **loader_kwargs)
    print("Using %d training features (%d batches) for class %s" % (len(current_dataset),
                                                                    len(training_loaders[decoder_class]),
                                                                    decoder_class),
          flush=True)

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

def augment(source_class, target_class):
    model.eval()

    # Process training dataset
    print("=> Processing training data...", flush=True)
    batches_processed = 0
    total_batches = len(training_loaders[source_class])
    with open(os.path.join(output_dir, "train-src_%s-tar_%s.ark" % (source_class, target_class)), 'w') as ark_fd:
        for batch_idx, (feats, targets, utt_ids) in enumerate(training_loaders[source_class]):
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
                frame_tensor = Variable(torch.FloatTensor(frame_spliced))
                if on_gpu:
                    frame_tensor = frame_tensor.cuda()

                recon_frames = model.forward_decoder(frame_tensor, target_class)
                recon_frames_numpy = recon_frames.cpu().data.numpy().reshape((-1, freq_dim))
                decoded_feats[i, :] = recon_frames_numpy[left_context:left_context + 1, :]

            # Write to output file
            write_kaldi_hao_ark(ark_fd, utt_id, decoded_feats)

            batches_processed += 1
            if batches_processed % log_interval == 0:
                print("===> Augmented %d/%d batches (%.1f%%)]" % (batches_processed,
                                                                  total_batches,
                                                                  100.0 * batches_processed / total_batches),
                      flush=True)

    # Create corresponding SCP file
    print("===> Writing SCP...", flush=True)
    with open(os.path.join(output_dir, "train-src_%s-tar_%s.scp" % (source_class, target_class)), 'w') as scp_fd:
        write_kaldi_hao_scp(scp_fd, os.path.join(output_dir, "train-src_%s-tar_%s.ark" % (source_class, target_class)))
    print("=> Done with training data", flush=True)

    # TODO: dev data


# Go through each combo of source and target class
for source_class in decoder_classes:
    for target_class in decoder_classes:
        print("PROCESSING SOURCE %s, TARGET %s" % (source_class, target_class), flush=True)
        augment(source_class, target_class)
print("Done data augmentation!", flush=True)
