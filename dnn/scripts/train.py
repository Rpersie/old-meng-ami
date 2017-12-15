import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, DataLoader

sys.path.append("../")
from dnn_md import DNNMultidecoder
from utils.kaldi_data import KaldiDataset

# Uses some structure from https://github.com/pytorch/examples/blob/master/vae/main.py

# Set up features
feature_name = "fbank"
feat_dim = int(os.environ["FEAT_DIM"])
left_splice = int(os.environ["LEFT_SPLICE"])
right_splice = int(os.environ["RIGHT_SPLICE"])
input_dim = (left_splice + right_splice + 1) * feat_dim

# Read in parameters to use for our network
enc_layer_sizes = []
for res_str in os.environ["ENC_LAYERS_DELIM"].split("_"):
    if len(res_str) > 0:
        enc_layer_sizes.append(int(res_str))
latent_dim = int(os.environ["LATENT_DIM"])
dec_layer_sizes = []
for res_str in os.environ["DEC_LAYERS_DELIM"].split("_"):
    if len(res_str) > 0:
        dec_layer_sizes.append(int(res_str))
activation = os.environ["ACTIVATION_FUNC"]
main_decoder_class = os.environ["MAIN_DECODER_CLASS"]
decoder_classes = []
for res_str in os.environ["DECODER_CLASSES_DELIM"].split("_"):
    if len(res_str) > 0:
        decoder_classes.append(res_str)

# Set up training parameters
batch_size = int(os.environ["BATCH_SIZE"])
epochs = int(os.environ["EPOCHS"])
optimizer_name = os.environ["OPTIMIZER"]
learning_rate = float(os.environ["LEARNING_RATE"])
on_gpu = torch.cuda.is_available()
log_interval = 1000   # Log results once for this many batches during training

# Set up data directories
training_dirs = dict()
training_dir_names = []
for res_str in os.environ["TRAIN_DIRS_DELIM"].split(" "):
    if len(res_str) > 0:
        training_dir_names.append(res_str)
for decoder_class in decoder_classes:
    for training_dir_name in training_dir_names:
        if decoder_class in training_dir_name:
            break
    training_dirs[decoder_class] = training_dir_name

dev_dirs = dict()
dev_dir_names = []
for res_str in os.environ["DEV_DIRS_DELIM"].split(" "):
    if len(res_str) > 0:
        dev_dir_names.append(res_str)
for decoder_class in decoder_classes:
    for dev_dir_name in dev_dir_names:
        if decoder_class in dev_dir_name:
            break
    dev_dirs[decoder_class] = dev_dir_name

# Fix random seed for debugging
torch.manual_seed(1)
if on_gpu:
    torch.cuda.manual_seed(1)

# Construct autoencoder with our parameters
print("Constructing model...", flush=True)
model = DNNMultidecoder(feat_dim=feat_dim,
                        splicing=[left_splice, right_splice], 
                        enc_layer_sizes=enc_layer_sizes,
                        latent_dim=latent_dim,
                        dec_layer_sizes=dec_layer_sizes,
                        activation=activation,
                        main_decoder_class=main_decoder_class,
                        decoder_classes=decoder_classes)
if on_gpu:
    model.cuda()
print("Done constructing model.", flush=True)
print(model, flush=True)

# Set up loss function
def reconstruction_loss(recon_x, x):
    MSE = nn.MSELoss()(recon_x, x.view(-1, input_dim))
    return MSE



# STEP 1: TRAIN AUTOENCODER ON MAIN DECODER CLASS



# Set up optimizer
wd = 5e-5
optimizer = getattr(optim, optimizer_name)(model.main_parameters(), lr=learning_rate, weight_decay=wd)

print("Setting up main data...", flush=True)
training_kaldi_filenames = list(filter(lambda x: "_splice" in x,
                                     filter(lambda x: feature_name in x,
                                            filter(lambda x: x.endswith(".scp"),
                                                   os.listdir(training_dirs[main_decoder_class])))))
training_kaldi_paths = list(map(lambda x: os.path.join(training_dirs[main_decoder_class], x), training_kaldi_filenames))
training_datasets = list(map(lambda x: KaldiDataset(x), training_kaldi_paths))
training_dataset = ConcatDataset(training_datasets)
print("Using %d training features" % len(training_dataset), flush=True)

dev_kaldi_filenames = list(filter(lambda x: "_splice" in x,
                                filter(lambda x: feature_name in x,
                                       filter(lambda x: x.endswith(".scp"),
                                              os.listdir(dev_dirs[main_decoder_class])))))
dev_kaldi_paths = list(map(lambda x: os.path.join(dev_dirs[main_decoder_class], x), dev_kaldi_filenames))
dev_datasets = list(map(lambda x: KaldiDataset(x), dev_kaldi_paths))
dev_dataset = ConcatDataset(dev_datasets)
print("Using %d dev features" % len(dev_dataset), flush=True)

loader_kwargs = {"num_workers": 1, "pin_memory": True} if on_gpu else {}
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

print("Done setting up main data.", flush=True)

def train(epoch, decoder_class):
    model.train()
    train_loss = 0.0
    for batch_idx, (feats, targets) in enumerate(training_loader):
        feats = Variable(feats)
        targets = Variable(targets)
        if on_gpu:
            feats = feats.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()
        recon_batch = model.forward_decoder(feats, decoder_class)
        loss = reconstruction_loss(recon_batch, targets)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % log_interval == 0:
            print("Train epoch %d: [%d/%d (%.1f%%)]\tLoss: %.6f" % (epoch,
                                                                    batch_idx * len(feats),
                                                                    len(training_loader.dataset),
                                                                    100.0 * batch_idx / len(training_loader),
                                                                    loss.data[0] / float(len(feats))),
                  flush=True)
    train_loss /= float(len(training_loader.dataset))
    return train_loss

def test(epoch, decoder_class):
    model.eval()
    test_loss = 0.0
    for feats, targets in dev_loader:
        if on_gpu:
            feats = feats.cuda()
            targets = targets.cuda()

        # Set to volatile so history isn't saved (i.e., not training time)
        feats = Variable(feats, volatile=True)
        targets = Variable(targets, volatile=True)
        recon_batch = model.forward_decoder(feats, decoder_class)
        test_loss += reconstruction_loss(recon_batch, targets).data[0]

    test_loss /= float(len(dev_loader.dataset))
    return test_loss

# Save model with best dev set loss thus far
best_dev_loss = float('inf')
save_best_only = True   # Set to False to always save model state, regardless of improvement

def save_checkpoint(state_obj, is_best, model_dir):
    filepath = os.path.join(model_dir, "ckpt_dnn_md_%s_%d.pth.tar" % (state_obj["decoder_class"], state_obj["epoch"]))
    torch.save(state_obj, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(model_dir, "best_dnn_md.pth.tar"))

# Regularize via patience-based early stopping
max_patience = 3
epochs_since_improvement = 0

# 1-indexed for pretty printing
print("Starting main training!", flush=True)
for epoch in range(1, epochs + 1):
    train_loss = train(epoch, main_decoder_class)
    print("====> Main Epoch %d: Average train loss %.6f" % (epoch, train_loss),
          flush=True)
    dev_loss = test(epoch, main_decoder_class)
    print("====> Main Dev set loss: %.6f" % dev_loss, flush=True)
    
    is_best = (dev_loss <= best_dev_loss)
    if is_best:
        best_dev_loss = dev_loss
        epochs_since_improvement = 0
        print("New best dev set loss: %.6f" % best_dev_loss, flush=True)
    else:
        epochs_since_improvement += 1
        print("No improvement in %d epochs (best dev set loss: %.6f)" % (epochs_since_improvement, best_dev_loss),
              flush=True)
        if epochs_since_improvement >= max_patience:
            print("STOPPING EARLY", flush=True)
            break

    if not (save_best_only and not is_best):
        # Save a checkpoint for our model!
        state_obj = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "best_dev_loss": best_dev_loss,
            "dev_loss": dev_loss,
            "optimizer": optimizer.state_dict(),
            "decoder_class": main_decoder_class
        }
        save_checkpoint(state_obj, is_best, os.environ["MODEL_DIR"])
        print("Saved checkpoint for model", flush=True)
    else:
        print("Not saving checkpoint; no improvement made", flush=True)



# STEP 2: TRAIN DECODERS FOR SECONDARY DECODER CLASSES



# Load checkpoint (potentially trained on GPU) into CPU memory (hence the map_location)
model_dir = os.environ["MODEL_DIR"]
print("Loading best checkpoint", flush=True)
checkpoint_path = "%s/best_dnn_md.pth.tar" % model_dir
checkpoint = torch.load(checkpoint_path, map_location=lambda storage,loc: storage)

# Set up model state and set to train mode (i.e. enable dropout)
model.load_state_dict(checkpoint["state_dict"])
model.train()
print("Loaded checkpoint; model ready now.")

for decoder_class in filter(lambda x: x != main_decoder_class, decoder_classes):
    # Set up optimizer
    optimizer = getattr(optim, optimizer_name)(model.decoder_parameters(decoder_class), lr=learning_rate)

    print("Setting up data for %s..." % decoder_class, flush=True)
    training_kaldi_filenames = list(filter(lambda x: "_splice" in x,
                                         filter(lambda x: feature_name in x,
                                                filter(lambda x: x.endswith(".scp"),
                                                       os.listdir(training_dirs[decoder_class])))))
    training_kaldi_paths = list(map(lambda x: os.path.join(training_dirs[decoder_class], x), training_kaldi_filenames))
    training_datasets = list(map(lambda x: KaldiDataset(x), training_kaldi_paths))
    training_dataset = ConcatDataset(training_datasets)
    print("Using %d training features for class %s" % (len(training_dataset), decoder_class), flush=True)

    dev_kaldi_filenames = list(filter(lambda x: "_splice" in x,
                                    filter(lambda x: feature_name in x,
                                           filter(lambda x: x.endswith(".scp"),
                                                  os.listdir(dev_dirs[decoder_class])))))
    dev_kaldi_paths = list(map(lambda x: os.path.join(dev_dirs[decoder_class], x), dev_kaldi_filenames))
    dev_datasets = list(map(lambda x: KaldiDataset(x), dev_kaldi_paths))
    dev_dataset = ConcatDataset(dev_datasets)
    print("Using %d dev features for class %s" % (len(dev_dataset), decoder_class), flush=True)

    loader_kwargs = {"num_workers": 1, "pin_memory": True} if on_gpu else {}
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    print("Done setting up data for %s." % decoder_class, flush=True)

    # Save model with best dev set loss thus far
    best_dev_loss = float('inf')
    save_best_only = True   # Set to False to always save model state, regardless of improvement

    # Regularize via patience-based early stopping
    max_patience = 3
    min_improvement = 0.001
    epochs_since_improvement = 0

    # 1-indexed for pretty printing
    print("Starting training for %s!" % decoder_class, flush=True)
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch, decoder_class)
        print("====> %s Epoch %d: Average train loss %.6f" % (decoder_class,
                                                              epoch,
                                                              train_loss),
              flush=True)
        dev_loss = test(epoch, decoder_class)
        print("====> %s Dev set loss: %.6f" % (decoder_class, dev_loss), flush=True)
        
        is_best = (best_dev_loss - dev_loss >= min_improvement)
        if is_best:
            best_dev_loss = dev_loss
            epochs_since_improvement = 0
            print("New best dev set loss: %.6f" % best_dev_loss, flush=True)
        else:
            epochs_since_improvement += 1
            print("No improvement in %d epochs (best dev set loss: %.6f)" % (epochs_since_improvement, best_dev_loss),
                  flush=True)
            if epochs_since_improvement >= max_patience:
                print("STOPPING EARLY", flush=True)
                break

        if not (save_best_only and not is_best):
            # Save a checkpoint for our model!
            state_obj = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_dev_loss": best_dev_loss,
                "dev_loss": dev_loss,
                "optimizer": optimizer.state_dict(),
                "decoder_class": decoder_class
            }
            save_checkpoint(state_obj, is_best, os.environ["MODEL_DIR"])
            print("Saved checkpoint for model", flush=True)
        else:
            print("Not saving checkpoint; no improvement made", flush=True)
