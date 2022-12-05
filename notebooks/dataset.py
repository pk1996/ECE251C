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
import argparse
from random import choice, randint
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from sphfile import SPHFile
import librosa

import torch
from torch.utils.data import Dataset, DataLoader
import pickle


BASE_DIR = os.path.join(os.getcwd(), '../')
DATA_DIR = os.path.join(os.getcwd(), '../data')


def compute_stft(audio_data, win_length=2048, hop_length=512, n_fft=2048):
    '''
    Helper method to compute the Short Time Fourier Transform
    '''
    return librosa.stft(audio_data, win_length=win_length, hop_length=hop_length, n_fft=n_fft)


def custom_collate_fn(batch):
    '''
    Custom collate function to batch spectrograms
    of diffenrent dimensions
    '''
    clean_spec = [torch.tensor(batch[i][1]) for i in range(len(batch))]
    noisy_spec = [torch.tensor(batch[i][0]) for i in range(len(batch))]

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


def pickle_stft_data(split, num_samples=1000, alpha=0.8, k=512):
    signal_len_lst = []

    noise_list = open(os.path.join(DATA_DIR, 'noise_list.txt')).readlines()
    file_list = open(os.path.join(DATA_DIR, '%s_set.txt'%(split))).readlines()

    clean_mixed_data_dict = {}
    clean_mixed_data_dict['clean'] = []
    clean_mixed_data_dict['mixed'] = []

    for itr in tqdm(range(num_samples)):
        sph_data = SPHFile(os.path.join(BASE_DIR, file_list[itr]).rstrip())
        samp_rate = sph_data.format['sample_rate']

        # Randomly sample noise sample from noise data list
        noise_data = librosa.load(os.path.join(BASE_DIR, choice(noise_list)).rstrip(), sr=samp_rate)
        assert(noise_data[1] == samp_rate == 16000)
        noise_signal = noise_data[0]
#         print(np.max(noise_signal), np.min(noise_signal))

        # Mixing noise with clean speech
        clean_signal = sph_data.content / (2**(sph_data.format['sample_sig_bits'] - 1))

        len_signal = min(clean_signal.shape[0], noise_signal.shape[0])
        signal_len_lst.append(len_signal)
#         print('Length of signal -- %d'%(len_signal))

        start_n = randint(0, max(0, noise_signal.shape[0] - clean_signal.shape[0]))

        # randomly sample a window from noise sequence to be mixed
        noise_signal = noise_signal[start_n:start_n+len_signal]
        clean_signal = clean_signal[0:len_signal]
        mixed_signal = alpha * clean_signal + (1-alpha) * noise_signal
#         print(np.max(mixed_signal), np.min(mixed_signal))
#         print('SNR -- %f' %(10*np.log10(alpha**2 * np.average(clean_signal**2)/((1-alpha)**2 * np.average((noise_signal)**2)))))

        stft_clean = compute_stft(clean_signal, win_length=k, n_fft=k)
        stft_mixed = compute_stft(mixed_signal, win_length=k, n_fft=k)

        clean_mixed_data_dict['clean'].append(stft_clean)
        clean_mixed_data_dict['mixed'].append(stft_mixed)
    
    with open(os.path.join(DATA_DIR, 'pkl_files/%s_data/clean_mixed_data_%d.pickle'%(split, num_samples)), 'wb') as handle:
        pickle.dump(clean_mixed_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return signal_len_lst

def pickle_stft_data_snr(split, num_samples=1000, alpha=0.8, k=512):
    '''
    Samples a random SNR in [15,20] and generates noisy signal accordingly.
    '''
    signal_len_lst = []

    noise_list = open(os.path.join(DATA_DIR, 'noise_list.txt')).readlines()
    file_list = open(os.path.join(DATA_DIR, '%s_set.txt'%(split))).readlines()

    clean_mixed_data_dict = {}
    clean_mixed_data_dict['clean'] = []
    clean_mixed_data_dict['mixed'] = []

    for itr in tqdm(range(num_samples)):
        snr = choice((15,20)) # randomly sample snr from 15-20
        sph_data = SPHFile(os.path.join(BASE_DIR, file_list[itr]).rstrip())
        samp_rate = sph_data.format['sample_rate']

        # Randomly sample noise sample from noise data list
        noise_data = librosa.load(os.path.join(BASE_DIR, choice(noise_list)).rstrip(), sr=samp_rate)
        assert(noise_data[1] == samp_rate == 16000)
        noise_signal = noise_data[0]
    #     print(np.max(noise_signal), np.min(noise_signal))

        # Mixing noise with clean speech
        clean_signal = sph_data.content / (2**(sph_data.format['sample_sig_bits'] - 1))

        len_signal = min(clean_signal.shape[0], noise_signal.shape[0])
        signal_len_lst.append(len_signal)
    #         print('Length of signal -- %d'%(len_signal))

        start_n = randint(0, max(0, noise_signal.shape[0] - clean_signal.shape[0]))

        # randomly sample a window from noise sequence to be mixed
        noise_signal = noise_signal[start_n:start_n+len_signal]
        clean_signal = clean_signal[0:len_signal]

        p_noise = np.average(noise_signal**2)
        p_signal = np.average(clean_signal**2)
        alpha = np.sqrt(p_signal/p_noise * 10**-(snr/10))

        mixed_signal = clean_signal + alpha * noise_signal
    #         print(np.max(mixed_signal), np.min(mixed_signal))
#         print('SNR -- %f' %(10*np.log10(np.average(clean_signal**2)/(alpha**2 * np.average((noise_signal)**2)))))

        stft_clean = compute_stft(clean_signal, win_length=k, n_fft=k)
        stft_mixed = compute_stft(mixed_signal, win_length=k, n_fft=k)

        clean_mixed_data_dict['clean'].append(stft_clean)
        clean_mixed_data_dict['mixed'].append(stft_mixed)

    with open(os.path.join(DATA_DIR, 'pkl_files/%s_data/clean_mixed_data_snr_%d.pickle'%(split, num_samples)), 'wb') as handle:
        pickle.dump(clean_mixed_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return signal_len_lst

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="specify options")
    parser.add_argument("split", type=str, choices=["train", "val", "test"], help="select split directory")
    parser.add_argument('--num_samples', type=int, default=1000, metavar='N', help="number of samples to save")
    args = parser.parse_args()

    signal_len_lst = pickle_stft_data(args.split, num_samples=args.num_samples)

    fig = plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(signal_len_lst, linestyle='', marker='.')
    plt.xlabel("file idx")
    plt.ylabel("signal length")
    plt.grid(linestyle='--')

    plt.subplot(1,2,2)
    counts, bins = np.histogram(signal_len_lst)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.xlabel("signal length")
    plt.ylabel("#signals")
    plt.grid(linestyle='--')
    
    plt.suptitle("%s data signal length for %d files"%(args.split, args.num_samples))

    plt.savefig(os.path.join(DATA_DIR, "pkl_files/%s_data/clean_mixed_data_%d.png"%(args.split, args.num_samples)))
    plt.show()
    
#     train_dataset = SpeechDataset('train')
    
#     # compute avg signal length
#     num_files = train_dataset.__len__()
#     print("Number of training files:", num_files)
#     print("Avg signal length (before):", train_dataset._avg_len / 10.0)
    
#     train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=custom_collate_fn)
    
#     start_time = time.time()
#     for i in tqdm(range(10)):
#         clean_spec, noisy_spec = next(iter(train_dataloader))
    
#         if i % 50 == 0:
#             print("{}/{}".format(i, 1000))
    
#     print("Time taken:", round(time.time() - start_time, 3), 's')
#     print("Avg signal length (after after):", train_dataset._avg_len / 10.0)