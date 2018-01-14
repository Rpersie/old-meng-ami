import bisect
import os
import struct

import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset

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
# Only for sequential use! Does not support random access
class HaoDataset(Dataset):
    def __init__(self, scp_path, left_context=0, right_context=0):
        super(HaoDataset, self).__init__()

        self.left_context = left_context
        self.right_context = right_context

        # Load in Hao files
        self.scp_path = scp_path
        self.scp_file = open(self.scp_path, 'r')

        # Determine how many utterances and features are included
        self.num_utts = 0
        self.num_feats = 0
        self.hao_ark_fd = None
        for scp_line in self.scp_file:
            self.num_utts += 1
            utt_id, feats, self.hao_ark_fd = read_next_utt(scp_line, hao_ark_fd=self.hao_ark_fd)
            self.num_feats += feats.shape[0]
        self.hao_ark_fd.close()
        self.hao_ark_fd = None

        # Track where we are with respect to feature index
        self.scp_file.seek(0)
        self.current_utt_id = None
        self.current_feat_mat = None
        self.current_feat_idx = 0

    def __del__(self):
        self.scp_file.close()
        if self.hao_ark_fd is not None:
            self.hao_ark_fd.close()
        
    def __len__(self):
        return self.num_feats

    def __getitem__(self, idx):
        if self.current_feat_mat is None:
            scp_line = self.scp_file.readline()
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
            
            self.current_feat_idx = 0

        feats_tensor = torch.FloatTensor(self.current_feat_mat[self.current_feat_idx:self.current_feat_idx + self.left_context + self.right_context + 1, :])
        feats_tensor = feats_tensor.view((-1, self.left_context + self.right_context + 1))

        # Target is identical to feature tensor
        target_tensor = feats_tensor.clone()

        # Update where we are in the feature matrix
        self.current_feat_idx += 1
        if self.current_feat_idx == len(self.current_feat_mat) - self.left_context - self.right_context:
            self.current_utt_id = None
            self.current_feat_mat = None
            self.current_feat_idx = 0

        if idx == len(self) - 1:
            # Reset back to the beginning
            self.scp_file.seek(0)

        return (feats_tensor, target_tensor)



# Utterance-by-utterance loading of Hao files
# Includes utterance ID data data for evaluation and decoding
class HaoEvalDataset(Dataset):
    def __init__(self, scp_path):
        super(HaoEvalDataset, self).__init__()

        # Load in Hao files
        self.scp_path = scp_path
        self.scp_file = open(self.scp_path, 'r')

        # Determine how many utterances are included
        self.utt_ids = []
        for scp_line in self.scp_file:
            utt_id, path_pos = scp_line.replace('\n','').split(' ')
            self.utt_ids.append(utt_id)
        self.num_utts = len(self.utt_ids)

        # Reset files
        self.scp_file.seek(0)

    # Utterance-level
    def __len__(self):
        return self.num_utts

    def utt_id(self, utt_idx):
        assert(utt_idx < len(self))
        return self.utt_ids[utt_idx]
    
    # Full utterance, not one frame
    def __getitem__(self, idx):
        # Get next utt from SCP file
        scp_line = self.scp_file.readline()
        utt_id, feat_mat, hao_ark_fd = read_next_utt(scp_line)
        feats_tensor = torch.FloatTensor(feat_mat)
        
        # Target is identical to feature tensor
        target_tensor = feats_tensor.clone()

        if idx == len(self) - 1:
            # Reset back to the beginning
            self.scp_file.seek(0)
        
        # Return utterance ID info as well
        return (feats_tensor, target_tensor, utt_id)



# Concat version of HaoEvalDataset
class HaoEvalConcatDataset(ConcatDataset):
    def utt_id(self, utt_idx):
        assert(utt_idx < len(self))
        dataset_idx = bisect.bisect_right(self.cummulative_sizes, utt_idx)
        if dataset_idx == 0:
            sample_idx = utt_idx
        else:
            sample_idx = utt_idx - self.cummulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].utt_id(sample_idx)
