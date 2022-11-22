'''
Dataset class for speech dataset.
returns clean and noisy spectorogram

1. Assumes data in 'data/ folder'
2. Hyperparameters - 
 a. alpha - to control mixing of clean speech and noise

TODO - Add support for stft parameterization if needed.

NOTE - see main() to create dataloader
'''

import os
import time
from random import choice, randint
from tqdm.notebook import tqdm

from sphfile import SPHFile
import librosa

import torch
from torch.utils.data import Dataset, DataLoader


def compute_stft(audio_data, win_length=2048, hop_length=512):
    '''
    Helper method to compute the Short Time Fourier Transform
    '''
    return librosa.stft(audio_data, win_length=win_length, hop_length=hop_length)


def custom_collate_fn(batch):
    '''
    Custom collate function to batch spectrograms
    of diffenrent dimensions
    '''
    clean_spec = [torch.tensor(batch[i][1]) for i in range(len(batch))]
    noisy_spec = [torch.tensor(batch[i][0]) for i in range(len(batch))]
    
#     for i in range(len(batch)):
#         clean_spec.append(torch.tensor(batch[i][1]))
#         noisy_spec.append(torch.tensor(batch[i][0]))
#         clean_spec.append(batch[i][1])
#         noisy_spec.append(batch[i][0])
    
#     clean_spec = torch.tensor(clean_spec)
#     noisy_spec = torch.tensor(noisy_spec)

    return clean_spec, noisy_spec


class SpeechDataset(Dataset):
    def __init__(self, split='train', alpha=0.8):
        assert split in ['train', 'val', 'test'], "Invalid split"
        self.split = split

        self.base_dir = os.path.join(os.getcwd(), '../')
        self.data_dir = os.path.join(os.getcwd(), '../data')
        self.noise_list = open(os.path.join(self.data_dir, 'noise_list.txt')).readlines()
        self.file_list = open(os.path.join(self.data_dir, '%s_set.txt'%(split))).readlines()
        self.alpha = alpha
        
        self._avg_len = 0
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        sph_data = SPHFile(os.path.join(self.base_dir, self.file_list[idx]).rstrip())
        samp_rate = sph_data.format['sample_rate']
        clean_signal = sph_data.content/(2**(sph_data.format['sample_sig_bits']-1))

        # Randomly sample noise sample from noise data list
        noise_signal = librosa.load(os.path.join(self.base_dir, choice(self.noise_list)).rstrip(), sr=samp_rate)[0]
                                  
        len_signal = min(clean_signal.shape[0], noise_signal.shape[0])
        self._avg_len += len_signal
        
        # randomly sample a window from noise sequence to be mixed
        start_n = randint(0, max(0, noise_signal.shape[0] - clean_signal.shape[0]))
        noise_signal = noise_signal[start_n:start_n+len_signal]
        clean_signal = clean_signal[0:len_signal]
        mixed_signal = self.alpha * clean_signal + (1 - self.alpha) * noise_signal
                                  
        # return STFT
        stft_mixed = compute_stft(mixed_signal, win_length=256)
        stft_clean = compute_stft(clean_signal, win_length=256)
#         print("STFT type:", stft_mixed.dtype, stft_clean.dtype)
        print("STFT size:", stft_mixed.shape, stft_clean.shape)
        
        return stft_mixed, stft_clean


if __name__ == '__main__':
    # To test dataloader
    train_dataset = SpeechDataset('train')
    
    # compute avg signal length
    num_files = train_dataset.__len__()
    print("Number of training files:", num_files)
    print("Avg signal length (before):", train_dataset._avg_len / 1000.0)
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=custom_collate_fn)
    
    for i in tqdm(range(5)):
        clean_spec, noisy_spec = next(iter(train_dataloader))
    
        # Print shape of spectorgram across a batch
        # print(itr[0][k].shape, itr[1][k].shape)
        if i % 50 == 0:
            print("{}/{}".format(i, 1000))
       
    print("Avg signal length (after after):", train_dataset._avg_len / 1000.0)