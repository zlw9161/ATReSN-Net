# -*- coding: utf-8 -*-
import random
import torch
import torch.utils.data as data
import numpy as np
import gc


class SpecAudioDataset(data.Dataset):

    def __init__(self, data_path, val_samples_per_audio, num_segs, transform=None, target_transform=None, mode='train'):
        r"""Simple data loader for spectrograms.
            Args:
                data_path (str): path to spectrogram dataset folder
                label_path (str): path to spectrogram label dictionary matching
            label ids to label names
                num_segs (int): number of segments for each audio.
                transform (object): set of augmentation steps defined by
            Compose(). Default is None.
                stack (bool): stack frames into a numpy.array. Default is True.
        """
        print('data loader')
        self.data_path = data_path
        self.val_samples = val_samples_per_audio
        self.num_segs = num_segs
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.class_names = []
        self.load_data()
        print('data loading --', len(self.data))

    def __getitem__(self, index):
        """
        With the given audio index, it fetches frames. This functions is called
        by Pytorch DataLoader threads. Each Dataloader thread loads a single
        batch by calling this function per instance.
        input sequence's shape is S*F*T*C (Seg_num * Time * Freq * Channel)
        output sequence's shape is C*S*F*T (Channel * Seg_num * Time * Freq)
        """ 
        sequence = self.data[index][0]
        label = self.data[index][1]
        audio_id = self.data[index][2]
        
        target = {'audio_id': audio_id,
                'label': label,
                'label_name': self.class_names[label]}
        if self.target_transform:
            target = self.target_transform(target)
        
        self.num_segs = sequence.shape[0]
        # normalization
        if self.transform:
            self.transform.randomize_parameters()
            sequence = [self.transform(seg) for seg in sequence]
            
        # format data to torch tensor
        sequence = torch.from_numpy(np.stack(sequence, 0).transpose(3, 0, 1, 2))

        return sequence, target

    def __len__(self):
        """
        This is called by PyTorch dataloader to decide the size of the dataset.
        """
        return len(self.data)

    def load_data(self):
        assert self.mode in ['train', 'val', 'test']
        self.data = []
        data_npz = np.load(self.data_path)
        self.class_names = data_npz['class_names']
        audio_ids = data_npz['audio_ids']
        if self.mode == 'train':
            num_samples = data_npz['y_train'].shape[0]
            labels = data_npz['y_train']
            feats = data_npz['X_train']
        if self.mode == 'val':
            num_samples = data_npz['y_val'].shape[0]
            labels = data_npz['y_val']
            feats = data_npz['X_val']
        if self.mode == 'test':
            num_samples = data_npz['y_test'].shape[0]
            labels = data_npz['y_test']
            feats = data_npz['X_test']
        for k in range(num_samples):
            self.data.append([feats[k], labels[k], audio_ids[k]])
        del feats
        gc.collect

    def multiply_data(self, n_times):
        self.data = self.data * n_times

