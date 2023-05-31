#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:14:35 2022

@author: mufassirin
"""
import pickle
import pandas as pd
import numpy as np
#import torch
import keras
import tensorflow as tf
# =============================================================================
# import pandas as pd
# 
# file_name = "/home/mufassirin/research/ss_predictor/dataset/My_Dataset/features/generated_features/"
# objects = pd.read_pickle(file_name)
# 
# =============================================================================

sample_prot = "1a0tP"
MISSING_LABEL = -99

objects = []
file_name = "/home/mufassirin/research/ss_predictor/dataset/My_Dataset/features/generated_features/1a0tP.pkl"

with (open(file_name, "rb")) as f:
    while True:
        try:
            objects.append(pickle.load(f))
        except EOFError:
            break
        
    
#======================================================================================     

def read_fasta_file(fname):
    """
    reads the sequence from the fasta file
    :param fname: filename (string)
    :return: protein sequence  (string)
    """
    with open(fname, 'r') as f:
        AA = ''.join(f.read().splitlines()[1:])
    return AA   

def one_hot(seq):
    """
    converts a sequence to one hot encoding
    :param seq: amino acid sequence (string)
    :return: one hot encoding of the amino acid (array)[L,20]
    """
    prot_seq = seq
    BASES = 'ARNDCQEGHILKMFPSTWYV'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
         in prot_seq])
    return feat
        
ss_conv_3_8_dict = {'X': 'X', 'C': 'C', 'S': 'C', 'T': 'C', 'H': 'H', 'G': 'H', 'I': 'H', 'E': 'E', 'B': 'E'}
SS3_CLASSES = 'CEH'
SS8_CLASSES = 'CSTHGIEB'

def read_ss3(fname, seq):
    """
    reads ss3 from .dssp file
     :params
         fname: filename (string)
         seq: protein sequence (string)
     :return
         ss3: (array) [L,3] (Int64)
    """
    ss3_labels = 'CEH'
    with open(fname, 'r') as f:
        ss8 = f.read().splitlines()[2]

    ss3_arr = np.array([ss_conv_3_8_dict[i] for i in ss8])[:, None]
    ss3_one_hot = (ss3_arr == np.array([i for i in ss3_labels])).astype(int)
    ss3_one_hot[ss3_one_hot.sum(1) == 0] = MISSING_LABEL
    if ss3_one_hot.shape[0] != len(seq):
        raise ValueError('Labels don''t match sequence length!')
    return ss3_one_hot


def read_ss8(fname, seq):
    """
    reads ss8 from .dssp file
     :params
         fname: filename (string)
         seq: protein sequence (string)
     :return
         ss8: (array) [L,8] (Int64)
    """
    ss8_labels = 'CSTHGIEB'
    with open(fname, 'r') as f:
        ss8 = f.read().splitlines()[2]
    ss8_arr = np.array([i for i in ss8])[:, None]
    ss8_one_hot = (ss8_arr == np.array([i for i in ss8_labels])).astype(int)
    ss8_one_hot[ss8_one_hot.sum(1) == 0] = MISSING_LABEL
    if ss8_one_hot.shape[0] != len(seq):
        raise ValueError('Labels don''t match sequence length!')
    return ss8_one_hot


if __name__ == "__main__":

    # read features
    fasta = read_fasta_file("/home/mufassirin/research/ss_predictor/dataset/My_Dataset/features/Train/"+ sample_prot +".fasta")
   # one_hot_seq = one_hot(fasta)
    #pssm = read_pssm("/home/mufassirin/Downloads/My_datasets/TEST2018/"+sample_prot+".pssm", fasta)
    #hmm = read_hhm("/home/mufassirin/Downloads/My_datasets/TEST2018/"+sample_prot+".hhm", fasta)

    # read_labels
    ss3 = read_ss3("/home/mufassirin/research/ss_predictor/dataset/My_Dataset/features/Train/"+sample_prot+".ss", fasta)
    ss8 = read_ss8("/home/mufassirin/research/ss_predictor/dataset/My_Dataset/features/Train/" + sample_prot + ".ss", fasta)
    
    print(len(fasta), ss3.shape, ss8.shape)
    print(objects)
    
    
       
    featuresd = np.concatenate([pssm[0], hmm[0]])

    for x in range(1,len(pssm)):
        featuresd = np.vstack([featuresd, np.concatenate([pssm[x], hmm[x]])])
    features_data = tf.constant(featuresd)
    #features_data = torch.tensor(featuresd)
    
    lables = np.concatenate([ss3[0], ss8[0]])
    for x in range(1,len(ss3)):
        lables = np.vstack([lables, np.concatenate([ss3[x], ss8[x]])])
    labels_data = tf.constant(lables)
    #labels_data = torch.tensor(lables)
