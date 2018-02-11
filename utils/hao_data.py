import bisect
import os
import struct

import numpy as np

import torch
from torch.utils.data import Dataset

def read_next_utt(scp_line, hao_ark_fd=None):
    # From https://github.com/yajiemiao/pdnn/blob/master/io_func/kaldi_feat.py
    if scp_line == '' or scp_line == None:
        return '', None
    utt_id, path_pos = scp_line.replace('\n','').split(' ')
    path, pos = path_pos.split(':')

    if hao_ark_fd is not None:
        if hao_ark_fd.name != path.split(os.sep)[-1]:
            # New hao_ark file now -- close and get new descriptor
            hao_ark_fd.close()
            hao_ark_fd = open(path, 'r')
    else:
        hao_ark_fd = open(path, 'r')
    hao_ark_fd.seek(int(pos),0)

    utt_id = hao_ark_fd.readline().rstrip('\n')

    tmp_mat = []
    current_line = hao_ark_fd.readline().rstrip('\n')
    while current_line != ".":
        tmp_mat.append(list(map(float, current_line.split(" "))))
        current_line = hao_ark_fd.readline().rstrip('\n')
    utt_mat = np.asarray(tmp_mat, dtype=np.float32)

    return utt_id, utt_mat, hao_ark_fd

def write_kaldi_hao_ark(hao_ark_fd, utt_id, arr):
    # From https://github.com/yajiemiao/pdnn/blob/master/io_func/kaldi_feat.py
    mat = np.asarray(arr, dtype=np.float32, order='C')
    rows, cols = mat.shape

    hao_ark_fd.write(utt_id + '\n')
    for row in range(mat.shape[0]):
        row_str = " ".join(map(str, mat[row, :]))
        hao_ark_fd.write(row_str + '\n')
    hao_ark_fd.write(".\n")

def write_kaldi_hao_scp(hao_scp_fd, hao_ark_filepath):
    with open(hao_ark_filepath, 'r') as ark_fd:
        last_line = ".\n"
        last_pos = 0
        
        for line in ark_fd:
            if last_line == ".\n":
                utt_id = line.rstrip('\n')
                hao_scp_fd.write("%s %s:%d\n" % (utt_id, hao_ark_filepath, last_pos))

            last_line = line
            last_pos += len(str.encode(line))   # Python 3 tell() doesn't work in text mode...

# Dataset class to support loading just features from Hao files
# Do not use Pytorch's built-in shuffle in DataLoader -- use the optional arguments here instead
class HaoDataset(Dataset):
    def __init__(self, scp_path, left_context=0, right_context=0, shuffle_utts=False, shuffle_feats=False):
        super(HaoDataset, self).__init__()

        self.left_context = left_context
        self.right_context = right_context

        # Load in Hao files
        self.scp_path = scp_path

        # Determine how many utterances and features are included
        self.num_utts = 0
        self.num_feats = 0
        self.hao_ark_fd = None
        self.scp_lines = []
        self.uttid_2_scpline = dict()
        with open(self.scp_path, 'r') as scp_file:
            for scp_line in scp_file:
                self.num_utts += 1
                utt_id, feats, self.hao_ark_fd = read_next_utt(scp_line, hao_ark_fd=self.hao_ark_fd)
                self.num_feats += feats.shape[0]
                self.scp_lines.append(scp_line)
                self.uttid_2_scpline[utt_id] = scp_line
        self.hao_ark_fd.close()
        self.hao_ark_fd = None
        
        # Set up shuffling of utterances within SCP (if enabled)
        self.shuffle_utts = shuffle_utts
        if self.shuffle_utts:
            # Shuffle SCP list in place
            np.random.shuffle(self.scp_lines)

        # Set up shuffling of frames within utterance (if enabled)
        self.shuffle_feats = shuffle_feats

        # Track where we are with respect to feature index
        self.current_utt_id = None
        self.current_feat_mat = None
        self.current_utt_idx = 0
        self.current_feat_idx = 0

    def __del__(self):
        if self.hao_ark_fd is not None:
            self.hao_ark_fd.close()
        
    def __len__(self):
        return self.num_feats

    def __getitem__(self, idx):
        if self.current_feat_mat is None:
            scp_line = self.scp_lines[self.current_utt_idx]
            self.current_utt_id, feat_mat, self.hao_ark_fd = read_next_utt(scp_line,
                                                                           hao_ark_fd=self.hao_ark_fd)

            # Duplicate frames at start and end of utterance (as in Kaldi)
            self.current_feat_mat = np.empty((feat_mat.shape[0] + self.left_context + self.right_context,
                                              feat_mat.shape[1]))
            self.current_feat_mat[self.left_context:self.left_context + feat_mat.shape[0], :] = feat_mat
            for i in range(self.left_context):
                self.current_feat_mat[i, :] = feat_mat[0, :]
            for i in range(self.right_context):
                self.current_feat_mat[self.left_context + feat_mat.shape[0] + i, :] = feat_mat[feat_mat.shape[0] - 1, :]

            # Shuffle features if enabled
            if self.shuffle_feats:
                np.random.shuffle(self.current_feat_mat)
            
            self.current_feat_idx = 0

        feats_tensor = torch.FloatTensor(self.current_feat_mat[self.current_feat_idx:self.current_feat_idx + self.left_context + self.right_context + 1, :])
        feats_tensor = feats_tensor.view((self.left_context + self.right_context + 1, -1))

        # Target is identical to feature tensor
        target_tensor = feats_tensor.clone()

        # Update where we are in the feature matrix
        self.current_feat_idx += 1
        if self.current_feat_idx == len(self.current_feat_mat) - self.left_context - self.right_context:
            self.current_utt_id = None
            self.current_feat_mat = None
            self.current_feat_idx = 0
            self.current_utt_idx += 1

        if idx == len(self) - 1:
            # We've seen all of the data (i.e. one epoch) -- shuffle SCP list in place
            np.random.shuffle(self.scp_lines)
            self.current_utt_idx = 0

        return (feats_tensor, target_tensor)

    # Get specific utterance 
    def feats_for_uttid(self, utt_id):
        utt_id, feat_mat, hao_ark_fd = read_next_utt(self.uttid_2_scpline[utt_id])
        return feat_mat



# Utterance-by-utterance loading of Hao files
# Includes utterance ID data data for evaluation and decoding
# Do not use Pytorch's built-in shuffle in DataLoader -- use the optional arguments here instead
class HaoEvalDataset(Dataset):
    def __init__(self, scp_path, shuffle_utts=False, shuffle_feats=False):
        super(HaoEvalDataset, self).__init__()

        # Load in Hao files
        self.scp_path = scp_path
        self.uttid_2_scpline = dict()
        with open(self.scp_path, 'r') as scp_file:
            # Determine how many utterances are included
            # Also sets up shuffling of utterances within SCP if desired
            self.scp_lines = list(map(lambda line: line.replace('\n', ''), scp_file.readlines()))
            self.utt_ids = []
            for scp_line in self.scp_lines:
                utt_id = scp_line.split(" ")[0]
                self.utt_ids.append(utt_id)
                self.uttid_2_scpline[utt_id] = scp_line
            
        self.shuffle_utts = shuffle_utts
        if self.shuffle_utts:
            # Shuffle SCP list in place
            np.random.shuffle(self.scp_lines)

        # Set up shuffling of feats within utterance
        self.shuffle_feats = shuffle_feats

    # Utterance-level
    def __len__(self):
        return len(self.utt_ids)

    def utt_id(self, utt_idx):
        assert(utt_idx < len(self))
        return self.utt_ids[utt_idx]
    
    # Get next utterance
    def __getitem__(self, idx):
        # Get next utt from SCP file
        scp_line = self.scp_lines[idx]
        utt_id, feat_mat, hao_ark_fd = read_next_utt(scp_line)
        if self.shuffle_feats:
            # Shuffle features in-place
            np.random.shuffle(feat_mat)
        feats_tensor = torch.FloatTensor(feat_mat)
        
        # Target is identical to feature tensor
        target_tensor = feats_tensor.clone()

        if idx == len(self) - 1 and self.shuffle_utts:
            # We've seen all of the data (i.e. one epoch) -- shuffle SCP list in place
            np.random.shuffle(self.scp_lines)
        
        # Return utterance ID info as well
        return (feats_tensor, target_tensor, utt_id)

    # Get specific utterance 
    def feats_for_uttid(self, utt_id):
        utt_id, feat_mat, hao_ark_fd = read_next_utt(self.uttid_2_scpline[utt_id])
        return feat_mat
