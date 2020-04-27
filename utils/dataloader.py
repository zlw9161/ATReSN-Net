
import os
import sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'dataloader_utils'))

import spec_transforms
import target_transforms
from datasets.SpecAudioDataset import SpecAudioDataset

def get_loader(root, train_transform, val_transform, target_transform, 
               batch_size=64, num_segs=8, val_samples=1, 
               n_threads=16, train_repeat=1, training=True, val=True, test=False):

    if training:
        # train dataset
        training_data = SpecAudioDataset(
            os.path.join(root, 'seq_diff_train.npz'),
            val_samples,
            num_segs,
            transform=train_transform,
            target_transform=target_transform,
            mode='train')
        if train_repeat > 1:
            training_data.multiply_data(train_repeat)
        # train loader
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_threads,
            sampler=None,
            drop_last=True)
    else:
        train_loader = None

    if val:
        # validation dataset
        validation_data = SpecAudioDataset(
            os.path.join(root, 'seq_diff_val.npz'),
            val_samples, 
            num_segs,
            transform=val_transform,
            target_transform=target_transform,
            mode='val')
        # val loader
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=batch_size // val_samples * val_samples,
            shuffle=False,
            num_workers=n_threads,
            drop_last=True)
    else:
        val_loader = None

    if test:
        # test dataset
        test_data = SpecAudioDataset(
            os.path.join(root, 'seq16_diff_test.npz'),
            #os.path.join(root, 'seq_diff_val.npz'),
            val_samples, 
            num_segs,
            transform=val_transform,
            target_transform=target_transform,
            mode='test')
        # test loader
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size // val_samples * val_samples,
            shuffle=False,
            num_workers=n_threads,
            drop_last=True)
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader




if __name__ == '__main__':
    # train normalization
    normalize = spec_transforms.ToNormalizedTensor()
    train_transform = spec_transforms.Compose([normalize])
    # validation normalization
    val_transform = spec_transforms.Compose([normalize])
    target_transform = target_transforms.ClassLabel()
    train_loader, val_loader = get_loader(root='/mnt/data/datasets/dcase2019/dev/task1a/',
                                          train_transform=train_transform, 
                                          val_transform=val_transform, 
                                          target_transform=target_transform,
                                          batch_size=64, num_segs=8, 
                                          val_samples=1, n_threads=16)
    print('run')
    begin_epoch = 0
    n_epochs = 1
    for epoch in range(begin_epoch, n_epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            clips = inputs.data.numpy()
            labels = targets.data.numpy()
            clip = clips[3]
            clip = np.transpose(clip, [1,2,3,0])
            print(labels)
            print(clips.shape)
