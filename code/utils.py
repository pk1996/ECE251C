import os
import yaml
import torch
import librosa

# Get mag and phase ----------------
def get_mag_phase(spec):
    mag = spec.real.type(torch.float) #torch.abs(spec).type(torch.float)
    phase = spec.imag.type(torch.float) #spec/mag
    return mag.unsqueeze(0).unsqueeze(0), phase.unsqueeze(0).unsqueeze(0)

#     mag = torch.abs(spec).type(torch.float)
#     phase = spec/mag
#     return mag.unsqueeze(0).unsqueeze(0), phase

# Config file handler ----------------
def load_config(config_name):
    CONFIG_PATH = '../configs'
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

# Custom collate function -------------
def custom_collate_fn(batch):
    '''
    Custom collate function to batch spectrograms
    of diffenrent dimensions
    '''
    clean_spec = [torch.tensor(batch[i][1]) for i in range(len(batch))]
    noisy_spec = [torch.tensor(batch[i][0]) for i in range(len(batch))]

    return clean_spec, noisy_spec

# Freq - time conversion -------------
def get_signal_from_spec(spec):
    '''
    Computes the time domain signal given the 
    pred and clean spectrogram
    '''
    return librosa.istft(spec.cpu().detach().numpy(), win_length=512, hop_length=512)