from pesq import pesq
from pystoi import stoi
import torch
import librosa
from tqdm import tqdm
import numpy as np
from utils import *

__all__ = ['evaluate', 'compute_snr', 'compute_stoi', 'compute_pesq', 'compute_spectrogram']


# --------------------------------
# Utility functions
# --------------------------------
def compute_snr(noisy_signal, clean_signal):
    '''
    Computes SNR given the noisy and clean 
    signals in time domain.
    '''
    return 10*np.log10(np.average(clean_signal**2)/np.average((noisy_signal-clean_signal)**2))

def compute_stoi(pred_signal, clean_signal):
    '''
    Computes the stoi given the noisy and clean
    signal in time domain
    '''
    return stoi(pred_signal, clean_signal, 16000)

def compute_pesq(pred_signal, clean_signal):
    '''
    Computes the pesq given the noisy and clean
    signal in time domain
    '''
    return pesq(16000, pred_signal, clean_signal)

def compute_spectrogram(spec_mag, spec_phase):
    '''
    Combine mag and phase to compute 
    the spectrogram
    '''
    return torch.multiply(spec_mag, spec_phase)

# --------------------------------
# Main eval function
# --------------------------------
def evaluate(model, test_dataloader, criterion, args):
    '''
    Evaluate trained model
    API design - TODO
    out = model(inp)
    eval_res = evaluate(out)
    '''
    print('Eval')
    eval_res = {}
    num_batches = len(test_dataloader)
    model.eval()     # evaluate model:
    
    LOSS = []
    STOI = []
    PESQ = []
    SNR  = []

    for itr in tqdm(range(num_batches)):
        clean_spec, noisy_spec = next(iter(test_dataloader))
        
        clean_spec = clean_spec[0].to(args['device'])
        noisy_spec = noisy_spec[0].to(args['device'])
        
        # compute the magnitude response of clean and noisy spectrograms
        clean_spec_mag, clean_spec_phase = get_mag_phase(clean_spec)
        noisy_spec_mag, noisy_spec_phase = get_mag_phase(noisy_spec)
        
        with torch.no_grad():
            # TODO - add __call__ in model.
            clean_spec_mag_pred, clean_spec_phase_pred = model(noisy_spec_mag, noisy_spec_phase)
            loss = criterion(clean_spec_mag_pred, clean_spec_phase_pred, clean_spec_mag, clean_spec_phase)
            
        # ISTFT for freq - time
        pred_spec = compute_spectrogram(clean_spec_mag_pred.squeeze().squeeze(), clean_spec_phase_pred.squeeze().squeeze())
#         pred_spec = torch.multiply(clean_spec_mag_pred.squeeze(0).squeeze(0), noisy_spec_phase)
        clean_signal = get_signal_from_spec(clean_spec)
        pred_signal = get_signal_from_spec(pred_spec)
        
        _snr = compute_snr(pred_signal, clean_signal)
        _stoi = compute_stoi(pred_signal, clean_signal)
        _pesq = compute_pesq(pred_signal, clean_signal)
             
        # Gather stats
        LOSS.append(loss)
        STOI.append(_stoi)
        SNR.append(_snr)
        PESQ.append(_pesq)
    
    eval_res['mse'] = LOSS
    eval_res['stoi'] = STOI
    eval_res['pesq'] = PESQ
    eval_res['snr'] = SNR
    
    return eval_res

if __name__ == '__main__':
    '''
    TODO - 
    Get model, test dataloader, args['device'].
    '''