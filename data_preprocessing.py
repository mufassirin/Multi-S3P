#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:43:43 2022

@author: mufassirin
"""

import os
from Bio import SeqIO
import numpy as np
import pickle


Features_DataPath = '/home/mufassirin/research/ss_predictor/dataset/My_Dataset/features/generated_features/test/CASP/npyForm'
Fasta_DataPath = '/home/mufassirin/research/ss_predictor/dataset/My_Dataset/features/test_sets/CASP/Fasta/'

HHM_DataPath = '/home/mufassirin/research/ss_predictor/dataset/My_Dataset/features/test_sets/CASP/HHM/'
PSSM_DataPath = '/home/mufassirin/research/ss_predictor/dataset/My_Dataset/features/test_sets/CASP/PSSM/'
PSP19_DataPath = '/home/mufassirin/research/ss_predictor/dataset/My_Dataset/features/test_sets/CASP/PSP19/'
Physico_DataPath = '/home/mufassirin/research/ss_predictor/dataset/My_Dataset/features/test_sets/CASP/7PCP/'


aa_index = { 'A' : -10, 'C' : -9, 'D' : -8, 'E' : -7, 'F' : -6, 'G' : -5,
               'H' : -4, 'I' : -3, 'K' : -2, 'L':-1, 'M' : 1, 'N' : 2,
                'P' : 3, 'Q' : 4, 'R' : 5, 'S' : 6, 'T' : 7, 'V' : 8,
               'W' : 9, 'Y' : 10}

def get_protein_ids_with_datapath(search_path, file_format):
    file_names = []
    files_path = []
    file_extension = file_format

    if os.path.exists(search_path):
        for (root, dirs, files) in os.walk(search_path):
            for file in files:
                if file.split('.')[-1] == file_extension: # e.g. file_extension='pdb', not '.pdb'
                #file_names.append(file.rstrip('.' + file_extension))
                    file_names.append(file[:-(len(file_extension)+1)]) # keeping file name only excluding extension and dot sign
                    files_path.append(root) # full path except file name
    return file_names

def extract_pssm(pssm_path,aa_dict):

	np_pssm_temp=np.zeros((0, 21))  
	with open(pssm_path) as p:
		lines = p.readlines()
		p.close()
	# print(lines)
	#print (protein_name)
      
	for line in lines:
		split_line = line.split()
		if (len(split_line) == 44) and (split_line[0] != '#'):
			aa_symbol = split_line[1]
			aa_number = aa_dict[aa_symbol]
			 
			pssm_temp = [-float(i) for i in split_line[2:22]]
			pssm_temp.insert(0, aa_number)
			vect_pssm_temp=np.array(pssm_temp)
			vect_pssm_temp=vect_pssm_temp.reshape(1,21)
			#print(vect_pssm_temp.shape)
			#print(type(vect_pssm_temp))
			np_pssm_temp = np.concatenate(( np_pssm_temp,vect_pssm_temp), axis=0)

	#row=np_pssm_temp.shape[0]
	return np_pssm_temp 

def extract_hhm(hhm_path):
    # for a single protein
    
    hhm_scores_all_residues = []
    seq_len = None
    with open(hhm_path) as hhm_file:
        lines = hhm_file.readlines()
        
        # Preprocess header - find seq_len and hhm score matrix's start location
        hhm_start_line_no = None
        for c_line_no, current_line in enumerate(lines):
            if current_line.startswith('LENG '):
                seq_len = int(current_line.split()[1])
            elif current_line.startswith('#\n') and lines[c_line_no+1].startswith('NULL ') and lines[c_line_no+2].startswith('HMM '):
                # HHM matrix starts after 3 consecutive lines starts wtih word '#', 'NULL' and 'HMM' Newline and. Space after these words is required as these may appear in other sentences
                # above these lines are meta information of HHm formation.
                hhm_start_line_no = c_line_no + 5     # HHM matrix header in 5 lines; skip thsese.
                break;
                
        # now for each of the residues there are three consecutive lines, read first one, skip next two lines
        if seq_len is not None and hhm_start_line_no is not None:
            residue_line_no = hhm_start_line_no
            for residue_no in range(seq_len):
                residual_fields = lines[residue_line_no].replace('*', '0').split() # first row - score for each individual residues; # replacing score '*' by '0'
                hhm_scores = [(-float(score))/1000 for score in residual_fields[2:22]] # 2:22 contains score for each amino acid (e.g. 20 standard amino acids)
                
                residue_line_no += 1
                residual_joint_fields = lines[residue_line_no].replace('*', '0').split() # 10 fields for pair score e.g. for M->M field
                hhm_scores.extend([(-float(score))/1000 for score in residual_joint_fields])
                
                hhm_scores = np.power(2, hhm_scores)                
                hhm_scores_all_residues.append(hhm_scores)
                
                residue_line_no += 2    # skip blank line, move to next residue start line
    return np.stack(hhm_scores_all_residues)


##main function
#protein_ids = np.loadtxt('/home/research/ss_predictor/CASP13.lst', dtype='str').tolist()
file_format = 'fasta' 
protein_ids = get_protein_ids_with_datapath(Fasta_DataPath, file_format)


i = 0
for protein_name in protein_ids:
    # Add sequence
    features = {}
    aa_seq_record = SeqIO.read(os.path.join(Fasta_DataPath,protein_name) +'.fasta', 'fasta')
    seq = str(aa_seq_record.seq)
    features['seq'] = seq

    
    # Add PSSM
    pssm_file = PSSM_DataPath + protein_name + ".pssm"
    pssm = extract_pssm(pssm_file,aa_index)
    features['pssm'] = pssm
    
    # Add HHM
    hhm_path = os.path.join(HHM_DataPath,protein_name) + '.hhm'
    hhm_scores = extract_hhm(hhm_path)
    features['hhm'] = hhm_scores
    
    
    ## Add Physicochemical encoding
    myphysico = np.load(os.path.join(Physico_DataPath,protein_name) + '.npy')
    assert myphysico.shape == (len(seq),7)
    features['physico'] = myphysico
    
    ## Add psp19 encoding
    mypsp19 = np.load(os.path.join(PSP19_DataPath,protein_name) + '.npy')
    assert mypsp19.shape == (len(seq),19)
    features['psp19'] = mypsp19
    
    # save
    pklfile = os.path.join(Features_DataPath,protein_name) +'.npy'
    f = open(pklfile, 'wb')
    pickle.dump(features,f)
    f.close()
    
    i = i+1
    print(i, protein_name, 'completed')
