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
from cnn_md import CNNAdversarialMultidecoder, CNNVariationalAdversarialMultidecoder
from utils.hao_data import HaoDataset

# Moved to function so that cProfile has a function to call
def run_training(run_mode, adversarial):
    run_start_t = time.clock()

    # Set up noising
    noise_ratio = float(os.environ["NOISE_RATIO"])
    print("Noising %.3f%% of input features" % (noise_ratio * 100.0), flush=True)

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

    model_dir = os.environ["MODEL_DIR"]
    if adversarial:
        best_ckpt_path = os.path.join(model_dir, "best_cnn_adversarial_fc_%s_act_%s_%s_ratio%s_md.pth.tar" % (os.environ["ADV_FC_DELIM"],
                                                                                                  adv_activation,
                                                                                                  run_mode,
                                                                                                  str(noise_ratio)))
    else:
        best_ckpt_path = os.path.join(model_dir, "best_cnn_%s_ratio%s_md.pth.tar" % (run_mode, str(noise_ratio)))
    print("Done constructing model.", flush=True)
    print(model, flush=True)

    # Count number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model has %d trainable parameters" % params, flush=True)

    # Set up loss functions
    def reconstruction_loss(recon_x, x):
        MSE = nn.MSELoss()(recon_x, x.view(-1, time_dim, freq_dim))
        return MSE

    def kld_loss(recon_x, x, mu, logvar):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= x.size()[0] * feat_dim

        return KLD
    
    def discriminative_loss(guess_class, truth_class):
        return nn.BCELoss()(guess_class, truth_class)



    # TRAIN MULTIDECODER



    # Set up optimizers for each decoder, as well as a shared encoder optimizer
    decoder_optimizers = dict()
    for decoder_class in decoder_classes:
        decoder_optimizers[decoder_class] = getattr(optim, optimizer_name)(model.decoder_parameters(decoder_class),
                                                                   lr=learning_rate)
    encoder_optimizer = getattr(optim, optimizer_name)(model.encoder_parameters(),
                                                       lr=learning_rate)
    if adversarial:
        adversary_optimizer = getattr(optim, optimizer_name)(model.adversary_parameters(),
                                                             lr=learning_rate)

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

    # Utilities for loss dictionaries
    def print_loss_dict(loss_dict, class_batches_processed):
        print("Losses:", flush=True)
        for decoder_class in loss_dict:
            print("=> Class %s" % decoder_class, flush=True)
            class_loss = 0.0
            for loss_key in loss_dict[decoder_class]:
                current_loss = loss_dict[decoder_class][loss_key] / class_batches_processed[decoder_class]
                class_loss += current_loss
                print("===> %s: %.3f" % (loss_key, current_loss), flush=True)
            print("===> Total for class %s: %.3f" % (decoder_class, class_loss), flush=True)
        print("TOTAL: %.3f" % total_loss(loss_dict, class_batches_processed), flush=True)

    def total_loss(loss_dict, class_batches_processed):
        loss = 0.0
        for decoder_class in loss_dict:
            for loss_key in loss_dict[decoder_class]:
                current_loss = loss_dict[decoder_class][loss_key] / class_batches_processed[decoder_class]
                loss += current_loss
        return loss



    def train(epoch):
        decoder_class_losses = {}
        for decoder_class in decoder_classes:
            decoder_class_losses[decoder_class] = {}

            decoder_class_losses[decoder_class]["autoencoding_recon_loss"] = 0.0
            decoder_class_losses[decoder_class]["backtranslation_recon_loss"] = 0.0
            if run_mode == "vae":
                # Track KL divergence as well
                decoder_class_losses[decoder_class]["autoencoding_kld"] = 0.0
                decoder_class_losses[decoder_class]["backtranslation_kld"] = 0.0
            if adversarial:
                decoder_class_losses[decoder_class]["adversarial_loss"] = 0.0

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
            
            # Add noise to signal; randomly drop out % of elements
            noise_matrix = torch.FloatTensor(np.random.binomial(1, 1.0 - noise_ratio, size=feats.size()).astype(float))
            noise_matrix = Variable(noise_matrix)
            if on_gpu:
                noise_matrix = noise_matrix.cuda()
            noised_feats = torch.mul(feats, noise_matrix)

            # PHASE 1: Backprop through same decoder (denoised autoencoding)
            model.train()
            encoder_optimizer.zero_grad()
            decoder_optimizers[decoder_class].zero_grad()
            if run_mode == "ae":
                recon_batch = model.forward_decoder(noised_feats, decoder_class)
            elif run_mode == "vae":
                recon_batch, mu, logvar = model.forward_decoder(noised_feats, decoder_class)
            else:
                print("Unknown train mode %s" % run_mode, flush=True)
                sys.exit(1)

            if run_mode == "ae":
                r_loss = reconstruction_loss(recon_batch, targets)
                r_loss.backward()
            elif run_mode == "vae":
                r_loss = reconstruction_loss(recon_batch, targets)
                k_loss = kld_loss(recon_batch, targets, mu, logvar)
                vae_loss = r_loss + k_loss
                vae_loss.backward()
            else:
                print("Unknown train mode %s" % run_mode, flush=True)
                sys.exit(1)

            decoder_class_losses[decoder_class]["autoencoding_recon_loss"] += r_loss.data[0]
            if run_mode == "vae":
                decoder_class_losses[decoder_class]["autoencoding_kld"] += k_loss.data[0]
            decoder_optimizers[decoder_class].step()
            encoder_optimizer.step()
            
            # PHASE 2: Backtranslation

            # Run (unnoised) features through other decoder in eval mode
            model.eval()
            if run_mode == "ae":
                translated_feats = model.forward_decoder(feats, other_decoder_class)
            elif run_mode == "vae":
                translated_feats, translated_mu, translated_logvar = model.forward_decoder(feats, other_decoder_class)
            else:
                print("Unknown train mode %s" % run_mode, flush=True)
                sys.exit(1)
            
            # Run translated features back through original decoder
            model.train()
            encoder_optimizer.zero_grad()
            decoder_optimizers[decoder_class].zero_grad()
            if run_mode == "ae":
                recon_batch = model.forward_decoder(translated_feats, decoder_class)
            elif run_mode == "vae":
                recon_batch, mu, logvar = model.forward_decoder(translated_feats, decoder_class)
            else:
                print("Unknown train mode %s" % run_mode, flush=True)
                sys.exit(1)

            if run_mode == "ae":
                r_loss = reconstruction_loss(recon_batch, targets)
                r_loss.backward()
            elif run_mode == "vae":
                r_loss = reconstruction_loss(recon_batch, targets)
                k_loss = kld_loss(recon_batch, targets, mu, logvar)
                vae_loss = r_loss + k_loss
                vae_loss.backward()
            else:
                print("Unknown train mode %s" % run_mode, flush=True)
                sys.exit(1)
            
            decoder_class_losses[decoder_class]["backtranslation_recon_loss"] += r_loss.data[0]
            if run_mode == "vae":
                decoder_class_losses[decoder_class]["backtranslation_kld"] += k_loss.data[0]
            decoder_optimizers[decoder_class].step()
            encoder_optimizer.step()
           
            if adversarial:
                # PHASE 3: Adversarial loss

                model.train()
                encoder_optimizer.zero_grad()
                adversary_optimizer.zero_grad()
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
                    print("Unknown train mode %s" % run_mode, flush=True)
                    sys.exit(1)
                
                class_prediction = model.adversary.forward(latent)
                class_truth = torch.FloatTensor(np.zeros(class_prediction.size())) if decoder_class == "ihm" else torch.FloatTensor(np.ones(class_prediction.size()))
                class_truth = Variable(class_truth)
                if on_gpu:
                    class_truth = class_truth.cuda()
                disc_loss = discriminative_loss(class_prediction, class_truth)

                # Train just discriminator
                old_adv_weights = model.state_dict()["adversary.lin_final.weight"].cpu().numpy()
                old_enc_weights = model.state_dict()["encoder_fc.lin_final.weight"].cpu().numpy()

                adversary_optimizer.zero_grad()
                disc_loss.backward(retain_graph=True)
                adversary_optimizer.step()
                new_adv_weights = model.state_dict()["adversary.lin_final.weight"].cpu().numpy()
                new_enc_weights = model.state_dict()["encoder_fc.lin_final.weight"].cpu().numpy()

                # Train just encoder, w/ negative discriminative loss
                old_adv_weights = model.state_dict()["adversary.lin_final.weight"].cpu().numpy()
                old_enc_weights = model.state_dict()["encoder_fc.lin_final.weight"].cpu().numpy()
                encoder_optimizer.zero_grad()
                adv_loss = -disc_loss
                adv_loss.backward()
                encoder_optimizer.step()
                new_adv_weights = model.state_dict()["adversary.lin_final.weight"].cpu().numpy()
                new_enc_weights = model.state_dict()["encoder_fc.lin_final.weight"].cpu().numpy()

                decoder_class_losses[decoder_class]["adversarial_loss"] += adv_loss.data[0]
            
            # Print updates, if any
            batches_processed += 1
            class_batches_processed[decoder_class] += 1
            if batches_processed % log_interval == 0:
                print("Train epoch %d: [%d/%d (%.1f%%)]" % (epoch,
                                                            batches_processed,
                                                            total_batches,
                                                            batches_processed / total_batches * 100.0),
                      flush=True)
                print_loss_dict(decoder_class_losses, class_batches_processed)

        return decoder_class_losses

    def test(epoch, loaders, recon_only=False, noised=True):
        decoder_class_losses = {}
        for decoder_class in decoder_classes:
            decoder_class_losses[decoder_class] = {}

            decoder_class_losses[decoder_class]["autoencoding_recon_loss"] = 0.0
            decoder_class_losses[decoder_class]["backtranslation_recon_loss"] = 0.0
            if run_mode == "vae" and not recon_only:
                # Track KL divergence as well
                decoder_class_losses[decoder_class]["autoencoding_kld"] = 0.0
                decoder_class_losses[decoder_class]["backtranslation_kld"] = 0.0
            if adversarial and not recon_only:
                decoder_class_losses[decoder_class]["adversarial_loss"] = 0.0

        other_decoder_class = decoder_classes[1]
        for decoder_class in decoder_classes:
            for feats, targets in loaders[decoder_class]:
                # Set to volatile so history isn't saved (i.e., not training time)
                feats = Variable(feats, volatile=True)
                targets = Variable(targets, volatile=True)
                if on_gpu:
                    feats = feats.cuda()
                    targets = targets.cuda()
            
                # Set up noising, if needed
                if noised:
                    # Add noise to signal; randomly drop out % of elements
                    noise_matrix = torch.FloatTensor(np.random.binomial(1, 1.0 - noise_ratio, size=feats.size()).astype(float))
                    noise_matrix = Variable(noise_matrix, volatile=True)
                    if on_gpu:
                        noise_matrix = noise_matrix.cuda()
                    noised_feats = torch.mul(feats, noise_matrix)

                # PHASE 1: Backprop through same decoder (denoised autoencoding)
                model.eval()
                if run_mode == "ae":
                    if noised:
                        recon_batch = model.forward_decoder(noised_feats, decoder_class)
                    else:
                        recon_batch = model.forward_decoder(feats, decoder_class)
                elif run_mode == "vae":
                    if noised:
                        recon_batch, mu, logvar = model.forward_decoder(noised_feats, decoder_class)
                    else:
                        recon_batch, mu, logvar = model.forward_decoder(feats, decoder_class)
                else:
                    print("Unknown train mode %s" % run_mode, flush=True)
                    sys.exit(1)

                if run_mode == "ae" or recon_only:
                    r_loss = reconstruction_loss(recon_batch, targets)
                elif run_mode == "vae":
                    r_loss = reconstruction_loss(recon_batch, targets)
                    k_loss = kld_loss(recon_batch, targets, mu, logvar)
                    vae_loss = r_loss + k_loss
                else:
                    print("Unknown train mode %s" % run_mode, flush=True)
                    sys.exit(1)

                decoder_class_losses[decoder_class]["autoencoding_recon_loss"] += r_loss.data[0]
                if run_mode == "vae" and not recon_only:
                    decoder_class_losses[decoder_class]["autoencoding_kld"] += k_loss.data[0]

                # PHASE 2: Backtranslation

                # Run (unnoised) features through other decoder
                model.eval()
                if run_mode == "ae":
                    translated_feats = model.forward_decoder(feats, other_decoder_class)
                elif run_mode == "vae":
                    translated_feats, translated_mu, translated_logvar = model.forward_decoder(feats, other_decoder_class)
                else:
                    print("Unknown train mode %s" % run_mode, flush=True)
                    sys.exit(1)
                
                # Run translated features back through original decoder
                model.eval()
                if run_mode == "ae":
                    recon_batch = model.forward_decoder(translated_feats, decoder_class)
                elif run_mode == "vae":
                    recon_batch, mu, logvar = model.forward_decoder(translated_feats, decoder_class)
                else:
                    print("Unknown train mode %s" % run_mode, flush=True)
                    sys.exit(1)

                if run_mode == "ae" or recon_only:
                    r_loss = reconstruction_loss(recon_batch, targets)
                elif run_mode == "vae":
                    r_loss = reconstruction_loss(recon_batch, targets)
                    k_loss = kld_loss(recon_batch, targets, mu, logvar)
                    vae_loss = r_loss + k_loss
                else:
                    print("Unknown train mode %s" % run_mode, flush=True)
                    sys.exit(1)
                
                decoder_class_losses[decoder_class]["backtranslation_recon_loss"] += r_loss.data[0]
                if run_mode == "vae" and not recon_only:
                    decoder_class_losses[decoder_class]["backtranslation_kld"] += k_loss.data[0]
               
                if adversarial and not recon_only:
                    # PHASE 3: Adversarial loss
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
                        print("Unknown train mode %s" % run_mode, flush=True)
                        sys.exit(1)
                    
                    class_prediction = model.adversary.forward(latent)
                    class_truth = torch.FloatTensor(np.zeros(class_prediction.size())) if decoder_class == "ihm" else torch.FloatTensor(np.ones(class_prediction.size()))
                    class_truth = Variable(class_truth, volatile=True)
                    if on_gpu:
                        class_truth = class_truth.cuda()
                    disc_loss = discriminative_loss(class_prediction, class_truth)
                    adv_loss = -disc_loss
                    
                    decoder_class_losses[decoder_class]["adversarial_loss"] += adv_loss.data[0]

            other_decoder_class = decoder_class
            
        return decoder_class_losses

    # Save model with best dev set loss thus far
    best_dev_loss = float('inf')
    save_best_only = True   # Set to False to always save model state, regardless of improvement

    def save_checkpoint(state_obj, is_best, model_dir):
        if not save_best_only:
            if adversarial:
                ckpt_path = os.path.join(model_dir, "ckpt_cnn_adversarial_fc_%s_act_%s_%s_ratio%s_md_%d.pth.tar" % (os.environ["ADV_FC_DELIM"],
                                                                                                        adv_activation,
                                                                                                        run_mode,
                                                                                                        str(noise_ratio),
                                                                                                        state_obj["epoch"]))
            else:
                ckpt_path = os.path.join(model_dir, "ckpt_cnn_%s_ratio%s_md_%d.pth.tar" % (run_mode, str(noise_ratio), state_obj["epoch"]))

            torch.save(state_obj, ckpt_path)
            if is_best:
                shutil.copyfile(ckpt_path, best_ckpt_path)
        else:
            torch.save(state_obj, best_ckpt_path)

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
        train_loss_dict = train(epoch)
        train_end_t = time.clock()
        print("\nEPOCH %d TRAIN (%.3fs)" % (epoch,
                                          train_end_t - train_start_t),
              flush=True)
        print_loss_dict(train_loss_dict, train_batch_counts)
            
        dev_start_t = time.clock()
        dev_loss_dict = test(epoch, dev_loaders)
        dev_end_t = time.clock()
        print("\nEPOCH %d DEV (%.3fs)" % (epoch,
                                        dev_end_t - dev_start_t),
              flush=True)
        print_loss_dict(dev_loss_dict, dev_batch_counts)
        dev_loss = total_loss(dev_loss_dict, dev_batch_counts)
        
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

        if not save_best_only or (save_best_only and is_best):
            # Save a checkpoint for our model!
            state_obj = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_dev_loss": best_dev_loss,
                "dev_loss": dev_loss,
                "decoder_optimizers": {decoder_class: decoder_optimizers[decoder_class].state_dict() for decoder_class in decoder_classes},
                "encoder_optimizer": encoder_optimizer.state_dict(),
            }
            if adversarial:
                state_obj["adversary_optimizer"] = adversary_optimizer.state_dict()

            save_checkpoint(state_obj, is_best, model_dir)
            print("Saved checkpoint for model", flush=True)
        else:
            print("Not saving checkpoint; no improvement made", flush=True)

    # Once done, load best checkpoint and determine reconstruction loss alone
    print("Computing reconstruction loss...", flush=True)

    # Load checkpoint (potentially trained on GPU) into CPU memory (hence the map_location)
    checkpoint = torch.load(best_ckpt_path, map_location=lambda storage,loc: storage)

    # Set up model state and set to eval mode (i.e. disable batch norm)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("Loaded checkpoint; best model ready now.")

    train_loss_dict = test(epoch, training_loaders, recon_only=True, noised=False)
    print("\nTRAINING SET", flush=True)
    print_loss_dict(train_loss_dict, train_batch_counts)

    dev_loss_dict = test(epoch, dev_loaders, recon_only=True, noised=False)
    print("\nDEV SET", flush=True)
    print_loss_dict(dev_loss_dict, dev_batch_counts)

    run_end_t = time.clock()
    print("\nCompleted training run in %.3f seconds" % (run_end_t - run_start_t), flush=True)



# Parse command line args
run_mode = "ae"
adversarial = False
profile = False

if len(sys.argv) >= 3:
    run_mode = sys.argv[1]
    adversarial = True if sys.argv[2] == "true" else False
    if len(sys.argv) >= 4:
        profile = True if sys.argv[3] == "profile" else False
else:
    print("Usage: python cnn/scripts/train_md.py <run mode> <adversarial true/false> <profile (optional)>", flush=True)
    sys.exit(1)

print("Running training with mode %s" % run_mode, flush=True)
if adversarial:
    print("Using adversarial loss", flush=True)
if profile:
    print("Profiling code using cProfile", flush=True)

    import cProfile
    profile_output_dir = os.path.join(os.environ["LOGS"], os.environ["EXPT_NAME"])
    profile_output_path = os.path.join(profile_output_dir, "train_%s.prof" % run_mode)
    cProfile.run('run_training(run_mode, adversarial)', profile_output_path) 
else:
    run_training(run_mode, adversarial)

