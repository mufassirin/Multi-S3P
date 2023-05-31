# Multi-S3P: Protein Secondary Structure Prediction with Specialized Multi-Network and Self-Attention-based Deep Learning Model

The standalone version of Multi-S3P is available for public use for research purposes.


## Table of Contents
- [Introduction](#introduction)
- [Dependency](#dependency)
- [Datasets](#datasets)
- [Features](#features)
- [Installation](#installation)
- [License](#license)

## Introduction

This study presents Multi-S3P, which employs bidirectional Long-Short-Term-Memory (BILSTM) and Convolutional Neural Networks (CNN) with a self-attention mechanism to improve the secondary structure prediction using an effective training strategy to capture the unique characteristics of each type of secondary structure and combine them more effectively. The ensemble of CNN and BILSTM can learn both contextual information and long-range interactions between the residues. In addition, using a self-attention mechanism allows the model to focus on the most important features for improving performance. We used the SPOT-1D dataset for the training and validation of our model using a set of four input features derived from amino acid sequences. Further, the model was tested on four popular independent test datasets and compared with various state-of-the-art predictors. The presented results show that Multi-S3P outperformed the other methods in terms of Q3, Q8 accuracy and other performance metrics, achieving the highest Q3 accuracy of 87.57\% and a Q8 accuracy of 77.56\% on the TEST2016 test set. More importantly, Multi-S3P demonstrates high performance in SS boundary regions. Our experiment also demonstrates that the combination of different input features and a multi-network-based training strategy significantly improved the performance.


## Dependency

- Python 3.7
- Anaconda
- TensorFlow v2.0

### Hardware Requirements: 
The Multi-S3P predictor underwent testing on a typical Ubuntu 18 computer with approximately 16 GB of RAM, ensuring sufficient memory capacity to facilitate in-memory operations.

## Datasets

### Train, Validation, TEST2016 & TEST2018 are obtained from SPOT-1D and available at: [Dataset](https://sparks-lab.org/server/spot-1d/) 
Cite: Hanson, J., Paliwal, K., Litfin, T., Yang, Y., & Zhou, Y. (2019). Improving prediction of protein secondary structure, backbone angles, solvent accessibility and contact numbers by using predicted contact maps and an ensemble of recurrent and residual convolutional neural networks. Bioinformatics, 35(14), 2403-2410.

CASP12 & CASP13: https://predictioncenter.org/

## Features

#### Input: (Total 76d)
- 20PSSM from PSI-BLAST v2.10.0 
- 30HMM from HHBlits v3.1.0
- 7PCP 
- 19PSP from [OPUS-PSP](https://www.sciencedirect.com/science/article/pii/S0022283607015045?casa_token=t78WkoWsEHcAAAAA:VRsI04nb9BRhs2gYtwcWw-mIesha-JxtrUnKnRrcsIbdoCrV7wjSaNppAiKBYH_YIsbq7azY2-c) 
- 
- Output: Secondary Structure Probabilities prot_name.csv

## Installation

#### Download and Unzip Multi-S3P source package.

1. ```$ git clone https://github.com/mufassirin/Multi-S3P.git ```
2. ``` $ cd Multi-S3P ```

### Download pretrained-model
- Please download the pretrained model weights from dropbox [here](https://www.dropbox.com/scl/fo/pu9omax6s30ttb98ddazt/h?dl=0&rlkey=7gsdy48yiyuuzinsj7oxbva52)
- Place it in the folder Multi-S3P. or

- ```wget https://www.dropbox.com/scl/fo/pu9omax6s30ttb98ddazt/h?dl=0&rlkey=7gsdy48yiyuuzinsj7oxbva52```



## License

MIT LICENSE

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Reference

