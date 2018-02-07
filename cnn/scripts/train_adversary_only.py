from collections import OrderedDict
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
from utils.hao_data import HaoDataset

def run_training(run_mode):
    run_start_t = time.clock()

    # Set up noising
    noise_ratio = float(os.environ["NOISE_RATIO"])
    print("Using model that noised %.3f%% of input features" % (noise_ratio * 100.0), flush=True)

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

    # Set up training parameters
    batch_size = int(os.environ["BATCH_SIZE"])
    epochs = int(os.environ["EPOCHS"])
    optimizer_name = os.environ["OPTIMIZER"]
    learning_rate = float(os.environ["LEARNING_RATE"])
    on_gpu = torch.cuda.is_available()
    log_interval = 1000   # Log results once for this many batches during training

    # Set up data files
    training_scps = dict()
    for decoder_class in decoder_classes:
        training_scp_name = os.path.join(os.environ["CURRENT_FEATS"], "%s-train-norm.blogmel.scp" % decoder_class)
        training_scps[decoder_class] = training_scp_name

    dev_scps = dict()
    for decoder_class in decoder_classes:
        dev_scp_name = os.path.join(os.environ["CURRENT_FEATS"], "%s-dev-norm.blogmel.scp" % decoder_class)
        dev_scps[decoder_class] = dev_scp_name

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
                                use_batch_norm=use_batch_norm,
                                decoder_classes=decoder_classes,
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
                                use_batch_norm=use_batch_norm,
                                decoder_classes=decoder_classes,
                                weight_init=weight_init)
    else:
        print("Unrecognized run mode %s" % run_mode, flush=True)
        sys.exit(1)

    if on_gpu:
        model.cuda()

    # Load checkpoint (potentially trained on GPU) into CPU memory (hence the map_location)
    print("Loading checkpoint...")
    model_dir = os.environ["MODEL_DIR"]
    best_ckpt_path = os.path.join(model_dir, "best_cnn_%s_ratio%s_md.pth.tar" % (run_mode, str(noise_ratio)))
    checkpoint = torch.load(best_ckpt_path, map_location=lambda storage,loc: storage)

    # Set up model state and set to eval mode (i.e. disable batch norm)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("Loaded checkpoint; best model ready now.")

    # Count number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model has %d trainable parameters" % params, flush=True)

    # Set up adversary to classify which domain the input came from, based on the latent vector
    # Simple 2-layer linear classifier
    adversary_layers = OrderedDict()
    '''
    if run_mode == "ae":
        adversary_layers["Linear_1"] = nn.Linear(latent_dim, 1024)
    elif run_mode == "vae":
        # Uses mu and logvar concatenated together
        adversary_layers["Linear_1"] = nn.Linear(2 * latent_dim, 1024)
    else:
        print("Unrecognized run mode %s" % run_mode, flush=True)
        sys.exit(1)
    adversary_layers["Sigmoid_1"] = nn.Sigmoid()
    adversary_layers["Linear_2"] = nn.Linear(1024, 1)
    adversary_layers["Sigmoid_2"] = nn.Sigmoid()
    '''
    adversary_layers["Linear_final"] = nn.Linear(latent_dim, 1)
    adversary_layers["Sigmoid_final"] = nn.Sigmoid()
    adversary = nn.Sequential(adversary_layers)
    if on_gpu:
        adversary = adversary.cuda()

    # Set up loss function
    def adversarial_loss(guess_class, truth_class):
        return nn.BCELoss()(guess_class, truth_class)



    # TRAIN ADVERSARY



    # Set up optimizer
    optimizer = optim.Adam(adversary.parameters(),
                           lr=0.0001)

    print("Setting up data...", flush=True)
    loader_kwargs = {"num_workers": 1, "pin_memory": True} if on_gpu else {}

    print("Setting up training datasets...", flush=True)
    training_datasets = dict()
    training_loaders = dict()
    for decoder_class in decoder_classes:
        current_dataset = HaoDataset(training_scps[decoder_class],
                                     left_context=left_context,
                                     right_context=right_context,
                                     shuffle_utts=True,
                                     shuffle_feats=True)
        training_datasets[decoder_class] = current_dataset
        training_loaders[decoder_class] = DataLoader(current_dataset,
                                                     batch_size=batch_size,
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
        current_dataset = HaoDataset(dev_scps[decoder_class],
                                     left_context=left_context,
                                     right_context=right_context,
                                     shuffle_utts=True,
                                     shuffle_feats=True)
        dev_datasets[decoder_class] = current_dataset
        dev_loaders[decoder_class] = DataLoader(current_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                **loader_kwargs)
        print("Using %d dev features (%d batches) for class %s" % (len(current_dataset),
                                                                   len(dev_loaders[decoder_class]),
                                                                   decoder_class),
              flush=True)

    # Set up minibatch shuffling (for training only) by decoder class
    print("Setting up minibatch shuffling for training...", flush=True)
    train_batch_counts = dict()
    total_batches = 0
    dev_batch_counts = dict()
    for decoder_class in decoder_classes:
        train_batch_counts[decoder_class] = len(training_loaders[decoder_class])
        total_batches += train_batch_counts[decoder_class]
        
        dev_batch_counts[decoder_class] = len(dev_loaders[decoder_class])
    print("%d total batches: %s" % (total_batches, str(train_batch_counts)), flush=True)

    print("Done setting up data.", flush=True)
    
    def train(epoch):
        train_loss = 0
        batches_processed = 0
        class_batches_processed = {decoder_class: 0 for decoder_class in decoder_classes}
        current_train_batch_counts = train_batch_counts.copy()
        training_iterators = {decoder_class: iter(training_loaders[decoder_class]) for decoder_class in decoder_classes}

        while batches_processed < total_batches:
            # Pick a decoder class
            batch_idx = random.randint(0, total_batches - batches_processed - 1)
            other_decoder_class = decoder_classes[1]
            for decoder_class in decoder_classes:
                current_count = current_train_batch_counts[decoder_class]
                if batch_idx < current_count:
                    current_train_batch_counts[decoder_class] -= 1
                    break
                else:
                    batch_idx -= current_count
                    other_decoder_class = decoder_class

            # Get data for this decoder class
            feats, targets = training_iterators[decoder_class].next()
            feats = Variable(feats)
            targets = Variable(targets)
            if on_gpu:
                feats = feats.cuda()
                targets = targets.cuda()

            adversary.train()
            model.eval()
            optimizer.zero_grad()

            if run_mode == "ae":
                latent, fc_input_size, unpool_sizes, pooling_indices = model.encode(feats.view(-1,
                                                                                               1,
                                                                                               time_dim,
                                                                                               freq_dim))
            elif run_mode == "vae":
                mu, logvar, fc_input_size, unpool_sizes, pooling_indices = model.encode(feats.view(-1,
                                                                                                   1,
                                                                                                   time_dim,
                                                                                                   freq_dim))
                latent = torch.cat((mu, logvar), 1)
            else:
                print("Unrecognized run mode %s" % run_mode, flush=True)
                sys.exit(1)

            class_prediction = adversary.forward(latent)

            class_truth = torch.FloatTensor(np.zeros(class_prediction.size())) if decoder_class == "ihm" else torch.FloatTensor(np.ones(class_prediction.size()))
            class_truth = Variable(class_truth)
            if on_gpu:
                class_truth = class_truth.cuda()
            loss = adversarial_loss(class_prediction, class_truth)
            loss.backward()

            train_loss += loss.data[0]
            optimizer.step()
            
            batches_processed += 1
            class_batches_processed[decoder_class] += 1
            if batches_processed % log_interval == 0:
                print("Train epoch %d: [%d/%d (%.1f%%)]" % (epoch,
                                                            batches_processed,
                                                            total_batches,
                                                            batches_processed / total_batches * 100.0),
                      flush=True)
                print("==> Loss: %.3f" % (train_loss / batches_processed), flush=True)

        return (train_loss / batches_processed)

    def test(epoch, loaders):
        test_loss = 0
        batches_processed = 0
        for decoder_class in decoder_classes:
            for feats, targets in loaders[decoder_class]:
                # Set to volatile so history isn't saved (i.e., not training time)
                feats = Variable(feats, volatile=True)
                targets = Variable(targets, volatile=True)
                if on_gpu:
                    feats = feats.cuda()
                    targets = targets.cuda()
            
                adversary.eval()
                model.eval()

                if run_mode == "ae":
                    latent, fc_input_size, unpool_sizes, pooling_indices = model.encode(feats.view(-1,
                                                                                                   1,
                                                                                                   time_dim,
                                                                                                   freq_dim))
                elif run_mode == "vae":
                    mu, logvar, fc_input_size, unpool_sizes, pooling_indices = model.encode(feats.view(-1,
                                                                                                       1,
                                                                                                       time_dim,
                                                                                                       freq_dim))
                    latent = torch.cat((mu, logvar), 1)
                else:
                    print("Unrecognized run mode %s" % run_mode, flush=True)
                    sys.exit(1)

                class_prediction = adversary.forward(latent)

                class_truth = torch.FloatTensor(np.zeros(class_prediction.size())) if decoder_class == "ihm" else torch.FloatTensor(np.ones(class_prediction.size()))
                class_truth = Variable(class_truth, volatile=True)
                if on_gpu:
                    class_truth = class_truth.cuda()
                loss = adversarial_loss(class_prediction, class_truth)

                test_loss += loss.data[0]
                batches_processed += 1
            
        return (test_loss / batches_processed)

    # Save model with best dev set loss thus far
    best_dev_loss = float('inf')

    # Regularize via patience-based early stopping
    max_patience = 3
    epochs_since_improvement = 0

    setup_end_t = time.clock()
    print("Completed setup in %.3f seconds" % (setup_end_t - run_start_t), flush=True)

    # 1-indexed for pretty printing
    print("Starting training!", flush=True)
    for epoch in range(1, epochs + 1):
        print("\nSTARTING EPOCH %d" % epoch, flush=True)

        train_start_t = time.clock()
        train_loss = train(epoch)
        train_end_t = time.clock()
        print("\nEPOCH %d TRAIN (%.3fs)" % (epoch,
                                          train_end_t - train_start_t),
              flush=True)
        print("===> TRAIN LOSS: %.3f" % train_loss, flush=True)
            
        dev_start_t = time.clock()
        dev_loss = test(epoch, dev_loaders)
        dev_end_t = time.clock()
        print("\nEPOCH %d DEV (%.3fs)" % (epoch,
                                        dev_end_t - dev_start_t),
              flush=True)
        print("===> DEV LOSS: %.3f" % dev_loss, flush=True)
        
        is_best = (dev_loss <= best_dev_loss)
        if is_best:
            best_dev_loss = dev_loss
            epochs_since_improvement = 0
            print("\nNew best dev set loss: %.6f" % best_dev_loss, flush=True)
        else:
            epochs_since_improvement += 1
            print("\nNo improvement in %d epochs (best dev set loss: %.6f)" % (epochs_since_improvement, best_dev_loss),
                  flush=True)
            if epochs_since_improvement >= max_patience:
                print("STOPPING EARLY", flush=True)
                break

    run_end_t = time.clock()
    print("\nCompleted training run in %.3f seconds" % (run_end_t - run_start_t), flush=True)



# Parse command line args
run_mode = "ae"
if len(sys.argv) >= 2:
    run_mode = sys.argv[1]
print("Running training with mode %s" % run_mode, flush=True)

run_training(run_mode)
