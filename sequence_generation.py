# -*- coding: utf-8 -*-
# @Author: Leon Zhang
# @Date: 2020/01/17
# Description: Get log-mel sequence for wave files.
# Dataset: DCASE-2019-Task1-ASC

import numpy as np
import pandas as pd
import librosa
import soundfile as sound

print("Librosa version = ",librosa.__version__)
print("Pysoundfile version = ",sound.__version__)

# File path
DataPath = '../data/dcase2019/Task1a/dev/'
TrainFile = DataPath + 'evaluation_setup/fold1_train.csv'
ValFile = DataPath + 'evaluation_setup/fold1_evaluate.csv'

# Audio info
sr = 48000
num_audio_channels = 2

# Duration unit is second
sample_duration = 10

# Log-Mel configuration
num_mel_banks = 128;
num_fft_points = 2048;
hop_length = int(num_fft_points/2)
num_frames = int(np.ceil(sample_duration*sr/hop_length))

# Segmentation configuration
seg_num = 8
seg_length = 80

# Load Wav filenames and labels
train_data_list = pd.read_csv(TrainFile, sep='\t', encoding='ASCII')
val_data_list = pd.read_csv(ValFile, sep='\t', encoding='ASCII')
train_wav_paths = train_data_list['filename'].tolist()
val_wav_paths = val_data_list['filename'].tolist()
train_labels = train_data_list['scene_label'].astype('category').cat.codes.values
val_labels = val_data_list['scene_label'].astype('category').cat.codes.values
class_names = np.unique(train_data_list['scene_label'])
num_classes = len(class_names)

# Calculate hop_frames for segmentation
def calculate_hop_frames(total_frames, segnum, seg_length):
    hop_frames = int((total_frames-seg_length)/(seg_num-1))
    return hop_frames

# Generate segment-level Log-Mel spectrograms and their deltas
# Training part
X_train = np.zeros((len(train_wav_paths),seg_num,num_mel_banks,seg_length,
                         num_audio_channels+1),'float32')
for i in range(len(train_wav_paths)):
    s, fs = sound.read(DataPath + train_wav_paths[i])
    s_diff = s[:,0] - s[:,1]
    s_diff = np.expand_dims(s_diff, -1)
    s = np.concatenate((s, s_diff), axis=-1)
    for channel in range(num_audio_channels+1):
        if len(s.shape)==1:
            s = np.expand_dims(s, -1)
        logmel = librosa.feature.melspectrogram(s[:,channel],
                                                sr = sr,
                                                n_fft = num_fft_points,
                                                hop_length = hop_length,
                                                n_mels = num_mel_banks,
                                                fmin = 0.0,
                                                fmax = sr/2,
                                                htk = True,
                                                norm = None)
        hop_frames = calculate_hop_frames(logmel.shape[1], seg_num, seg_length)
        for seg in range(seg_num):
            X_train[i,seg,:,:,channel] = logmel[:,
                        (seg*hop_frames):(seg*hop_frames+seg_length)]
        print('Generating Log-Mel for training wav file #%d complete!'%(i))
X_train = np.log(X_train + 1e-8)

# Validation part
X_val = np.zeros((len(val_wav_paths),seg_num,num_mel_banks,seg_length,
                       num_audio_channels+1),'float32')
for i in range(len(val_wav_paths)):
    s, fs = sound.read(DataPath + val_wav_paths[i])
    s_diff = s[:,0] - s[:,1]
    s_diff = np.expand_dims(s_diff, -1)
    s = np.concatenate((s, s_diff), axis=-1)
    for channel in range(num_audio_channels+1):
        logmel = librosa.feature.melspectrogram(s[:,channel],
                                                sr = sr,
                                                n_fft = num_fft_points,
                                                hop_length = hop_length,
                                                n_mels = num_mel_banks,
                                                fmin = 0.0,
                                                fmax = sr/2,
                                                htk = True,
                                                norm = None)
        hop_frames = calculate_hop_frames(logmel.shape[1], seg_num, seg_length)
        for seg in range(seg_num):
            X_val[i,seg,:,:,channel] = logmel[:,
                      (seg*hop_frames):(seg*hop_frames+seg_length)]
        print('Generating Log-Mel for validation wav file #%d complete!'%(i))
X_val = np.log(X_val + 1e-8)


# Get segment-level one hot labels
#y_train = np.zeros((len(train_wav_paths)*seg_num, num_classes),'int')
y_train = np.zeros((len(train_wav_paths)),'int')
for i in range(len(train_wav_paths)):
    y_train[i] = int(train_labels[i])

#y_val = np.zeros((len(val_wav_paths)*seg_num, num_classes),'int')
y_val = np.zeros((len(val_wav_paths)),'int')
for i in range(len(val_wav_paths)):
    y_val[i] = int(val_labels[i])

# Save log-mels and labels
np.savez(DataPath+'seq_diff_train.npz', X_train = X_train, y_train = y_train,
         audio_ids = train_wav_paths,
         class_names = class_names)
np.savez(DataPath+'seq_diff_val.npz', X_val = X_val, y_val = y_val,
         audio_ids = val_wav_paths,
         class_names = class_names)
