# ATReSN-Net
Capturing attentive temporal relations in semantic neighborhood for ASC (DCASE 2018/2019 Task 1a)

If you find our work useful in your research, please cite:

        @article{Zhang:2020:atresn-net,
          title={ATReSN-Net: Capturing Attentive Temporal Relations in Semantic Neighborhood for Acoustic Scene Classification},
          author={Liwen Zhang, Jiqing Han and Ziqiang Shi},
          journal={Ready for submission of IS2020},
          year={2020}
        }

## Setup
1. tensorflow > 1.9.0
2. dataflow --> tensorpack 
link: https://github.com/tensorpack

## Description

### 1. Training/testing/evaluating sh scripts:
usage: sh command_train(evaluate/test)_dcase2019.sh

### 2. Data preparation:
Use sequence_generation.py to produce Log-Mel spec sequence for each audio wav data.
The ./utils/dataloader.py script is used to load the generated sequence into the network. It is implemented by using the interface "datasets" of tensorpack. The dataset class declarition is in ./utils/datasets/SpecAudioDataset.py.

### 3. Pretrained models:
The pretrained models are in the ./pretrained_models. We modified the example of ResNet for ImageNet in https://github.com/tensorpack/tensorpack/tree/master/examples/ResNet to get the pretrained 2D ResNet-18 and PreAct-18.

### 4. Transform ResNet-18 into ATReSN-Net:
The ATReSN-Net definition is in ./models/resnet18_atresn_128_80.py, which will load the pre-trained models and transform them into ATReSN-Nets by calling the methods in ./tf_utils.py.

### 5. ATReSN module:
(1) Semantic Neighborhood Grouper (SNG) is worked as an operation in the ATReSN-Net. The codes are in the ./tf_ops.
This operation is modified from the CPNet in https://github.com/xingyul/cpnet.
(2) Attentive Temporal Relations Aggregator (ATRA) is used for aggregating temporal relations from the neighborhood grouped by SNG, it is based on the attentive pooling, which is inspired by the RandLA-Net in https://github.com/QingyongHu/RandLA-Net.

### 6. Train/test the model:
Use ./train.py and ./test.py to train and test the model.

### 7. References
* <a href="https://ieeexplore.ieee.org/document/8960462" target="_blank">Pyramidal Temporal Pooling With Discriminative Mapping for Audio Classification
</a> by Zhang et al. (IEEE Trans. ASLP). Code and data released in <a href="https://github.com/zlw9161/PyramidalTemporalPooling">GitHub</a>.
* <a href="http://arxiv.org/abs/1905.07853" target="_blank">Learning Video Representations from Correspondence Proposals
</a> by Liu et al. (CVPR 2019). Code and data released in <a href="https://github.com/xingyul/cpnet">GitHub</a>.
* <a href="http://stanford.edu/~rqi/pointnet" target="_blank">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017 Oral Presentation). Code and data released in <a href="https://github.com/charlesq34/pointnet">GitHub</a>.
* <a href="https://arxiv.org/abs/1911.11236?context=cs.LG" target="_blank">RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds</a> by Hu et al. (CVPR 2020 Oral Presentation). Code and data released in <a href="https://github.com/QingyongHu/RandLA-Net">GitHub</a>.

### 8. Related work
Learning temporal relations from semantic neighbors for Acoustic Scene Classification </a> by Zhang et al. (Submitted to IEEE SPL on Mar 10th, 2020). Code and data released in <a href="https://github.com/zlw9161/SeNoT-Net">GitHub</a>.
