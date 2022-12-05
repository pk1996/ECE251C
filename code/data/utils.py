import os
import librosa
import pickle
from sphfile import SPHFile
from pathlib import Path
from tqdm import tqdm
from random import choice, randint
import numpy as np

BASE_DIR = Path(os.getcwd()).parent
DATA_DIR = BASE_DIR / 'data'

# STFT ---------------------
def compute_stft(audio_data, win_length=2048, hop_length=512, n_fft=2048):
    '''
    Helper method to compute the Short Time Fourier Transform
    '''
    return librosa.stft(audio_data, win_length=win_length, hop_length=hop_length, n_fft=n_fft)


# Generate pickle (alpha based) ---------------------
def pickle_stft_data(split, num_samples=1000, k=512):
    '''
    Generates noisy speech data by -
    noisy = alpha * clean + (1-alpha)*noise
    '''
    alpha = choice((6,9))/10 # random alpha
    signal_len_lst = []

    noise_list = open(os.path.join(DATA_DIR, 'noise_list.txt')).readlines()
    file_list = open(os.path.join(DATA_DIR, '%s_set.txt'%(split))).readlines()

    clean_mixed_data_dict = {}
    clean_mixed_data_dict['clean'] = []
    clean_mixed_data_dict['mixed'] = []
    
    pbar = tqdm(total = num_samples, leave=True, position=0) # show update
    itr = 0
    
    while len(signal_len_lst) < num_samples and itr < len(file_list): #itr in tqdm(range(num_samples)):
        sph_data = SPHFile(os.path.join(BASE_DIR, file_list[itr]).rstrip())
        itr += 1
        samp_rate = sph_data.format['sample_rate']

        # Randomly sample noise sample from noise data list
        noise_data = librosa.load(os.path.join(BASE_DIR, choice(noise_list)).rstrip(), sr=samp_rate)
        assert(noise_data[1] == samp_rate == 16000)
        noise_signal = noise_data[0]

        # Mixing noise with clean speech
        clean_signal = sph_data.content / (2**(sph_data.format['sample_sig_bits'] - 1))

        len_signal = min(clean_signal.shape[0], noise_signal.shape[0])
        
        # ignore data if spec size < 64
        if len_signal/k < 64: continue
        pbar.update(1)
        
        # limit signal length to k*64
        len_signal = k*64-1
        
        signal_len_lst.append(len_signal)
        start_n = randint(0, max(0, noise_signal.shape[0] - clean_signal.shape[0]))

        # randomly sample a window from noise sequence to be mixed
        noise_signal = noise_signal[start_n:start_n+len_signal]
        clean_signal = clean_signal[0:len_signal]
        mixed_signal = alpha * clean_signal + (1-alpha) * noise_signal
        
        stft_clean = compute_stft(clean_signal, win_length=k, n_fft=k)
        stft_mixed = compute_stft(mixed_signal, win_length=k, n_fft=k)

        clean_mixed_data_dict['clean'].append(stft_clean)
        clean_mixed_data_dict['mixed'].append(stft_mixed)
    
    with open(os.path.join(DATA_DIR, 'pkl_files/%s_data/clean_mixed_data_%d.pickle'%(split, num_samples)), 'wb') as handle:
        pickle.dump(clean_mixed_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return signal_len_lst

# # Generate pickle (SNR-based) ---------------------
# def pickle_stft_data_snr(split, num_samples=1000, k=512):
#     '''
#     Samples a random SNR in [15,20] and generates noisy signal accordingly.
#     '''
#     signal_len_lst = []

#     noise_list = open(os.path.join(DATA_DIR, 'noise_list.txt')).readlines()
#     file_list = open(os.path.join(DATA_DIR, '%s_set.txt'%(split))).readlines()

#     clean_mixed_data_dict = {}
#     clean_mixed_data_dict['clean'] = []
#     clean_mixed_data_dict['mixed'] = []
    
#     pbar = tqdm(total = num_samples, leave=True, position=0) # show update
#     itr = 0
    
#     while len(signal_len_lst) < num_samples and itr < len(file_list): #itr in tqdm(range(num_samples)):
#         snr = choice((15,20)) # randomly sample snr from 15-20
#         sph_data = SPHFile(os.path.join(BASE_DIR, file_list[itr]).rstrip())
#         samp_rate = sph_data.format['sample_rate']

#         # Randomly sample noise sample from noise data list
#         noise_data = librosa.load(os.path.join(BASE_DIR, choice(noise_list)).rstrip(), sr=samp_rate)
#         assert(noise_data[1] == samp_rate == 16000)
#         noise_signal = noise_data[0]

#         # Mixing noise with clean speech
#         clean_signal = sph_data.content / (2**(sph_data.format['sample_sig_bits'] - 1))

#         len_signal = min(clean_signal.shape[0], noise_signal.shape[0])
        
#         # ignore data if spec size < 64
#         if len_signal/k < 64: continue
#         pbar.update(1)
        
#         # limit signal length to k*64
#         len_signal = k*64
        
#         signal_len_lst.append(len_signal)
#         start_n = randint(0, max(0, noise_signal.shape[0] - clean_signal.shape[0]))

#         # randomly sample a window from noise sequence to be mixed
#         noise_signal = noise_signal[start_n:start_n+len_signal]
#         clean_signal = clean_signal[0:len_signal]

#         p_noise = np.average(noise_signal**2)
#         p_signal = np.average(clean_signal**2)
#         alpha = np.sqrt(p_signal/p_noise * 10**-(snr/10))

#         mixed_signal = clean_signal + alpha * noise_signal

#         stft_clean = compute_stft(clean_signal, win_length=k, n_fft=k)
#         stft_mixed = compute_stft(mixed_signal, win_length=k, n_fft=k)

#         clean_mixed_data_dict['clean'].append(stft_clean)
#         clean_mixed_data_dict['mixed'].append(stft_mixed)

#     with open(os.path.join(DATA_DIR, 'pkl_files/%s_data/clean_mixed_data_snr_%d.pickle'%(split, num_samples)), 'wb') as handle:
#         pickle.dump(clean_mixed_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#     return signal_len_lst

# Load data from pkl file ---------------------
def load_pkl_data(split, num_samples):
    pkl_file = DATA_DIR / 'pkl_files' / '{}_data'.format(split) / 'clean_mixed_data_{}.pickle'.format(num_samples)
    print('Reading from %s \n'%(pkl_file))
    if not pkl_file.is_file():
        print('Generating data ...')
        signal_len_lst = pickle_stft_data(split, num_samples=num_samples)
        
    with open(pkl_file, 'rb') as handle:
        clean_mixed_data_dict = pickle.load(handle)
    
    return clean_mixed_data_dict
