"""
Implements model1 which only cleans the magnitude spectrum
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_util import *

class ConvRecNet_Complex(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output:[batch size, channels=1, T, n_fft]
    """
    
    def __init__(self, params):
        super(ConvRecNet_Complex, self).__init__()
        
        # Encoder
        self.conv_block_1 = CausalConvBlock(1, 16, params)
        self.conv_block_2 = CausalConvBlock(16, 32, params)
        self.conv_block_3 = CausalConvBlock(32, 64, params)
        self.conv_block_4 = CausalConvBlock(64, 128, params)
        self.conv_block_5 = CausalConvBlock(128, 256, params)
        
        # LSTM
        self.lstm_layer = nn.LSTM(input_size=256*9, hidden_size=256*9, num_layers=2, batch_first=True)
        
        # decoder for magnitude
        self.decoder_m = nn.ModuleList()
        self.decoder_m.append(CausalTransConvBlock(256 + 256, 128))
        self.decoder_m.append(CausalTransConvBlock(128 + 128, 64))
        self.decoder_m.append(CausalTransConvBlock(64 + 64, 32))
        self.decoder_m.append(CausalTransConvBlock(32 + 32, 16))
        self.decoder_m.append(CausalTransConvBlock(16 + 16, 1, is_last=True))
        
        # decoder for phase
        self.decoder_p = nn.ModuleList()
        self.decoder_p.append(CausalTransConvBlock(256 + 256, 128))
        self.decoder_p.append(CausalTransConvBlock(128 + 128, 64))
        self.decoder_p.append(CausalTransConvBlock(64 + 64, 32))
        self.decoder_p.append(CausalTransConvBlock(32 + 32, 16))
        self.decoder_p.append(CausalTransConvBlock(16 + 16, 1, is_last=True))

    def encoder(self, x):
        '''
        Common encoder for phase and mag spec
        '''
        self.lstm_layer.flatten_parameters()
        e1 = self.conv_block_1(x)
        e2 = self.conv_block_2(e1)
        e3 = self.conv_block_3(e2)
        e4 = self.conv_block_4(e3)
        e5 = self.conv_block_5(e4)  # [2, 256, 4, 200]
        return e1, e2, e3, e4, e5
        
    def lstm_(self, x, batch_size, n_channels, n_f_bins, n_frame_size):
        '''
        Common LSTM block
        '''
        lstm_in = x.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
        lstm_out, _ = self.lstm_layer(lstm_in)  # [2, 200, 1024]
        lstm_out = lstm_out.permute(0, 2, 1).reshape(batch_size, n_channels, n_f_bins, n_frame_size)  # [2, 256, 4, 200]
        return lstm_out
    
    def decoder(self, lstm_out, e1, e2, e3, e4, e5, isMag = True):
        '''
        decoder for mag spec
        '''
        encoder_op = [e5, e4, e3 , e2, e1]
        out = [lstm_out]
        
        if isMag:
            for i, decoder_layer in enumerate(self.decoder_m):
                out.append(decoder_layer(torch.cat((out[-1], encoder_op[i]), 1)))
        else:
            for i, decoder_layer in enumerate(self.decoder_p):
                out.append(decoder_layer(torch.cat((out[-1], encoder_op[i]), 1)))
                
        return out[-1]
        
    def forward(self, mag, phase):
        """
        Input - Noisy magnitude and phase spectrums
        Output - Cleaned up magnitude and phase spectrums
        
        Note - This model doesn't do anything with the phase spectrum
        """
        mag, pad = Padding.pad(mag, 32)
        phase, pad = Padding.pad(phase, 32)
        
        # Encoder
        em1, em2, em3, em4, em5 = self.encoder(mag)
        ep1, ep2, ep3, ep4, ep5 = self.encoder(phase)
        
        # LSTM
        self.lstm_layer.flatten_parameters()
        batch_size, n_channels, n_f_bins, n_frame_size = em5.shape
        
        lstm_out_m = self.lstm_(em5, batch_size, n_channels, n_f_bins, n_frame_size)
        lstm_out_p = self.lstm_(ep5, batch_size, n_channels, n_f_bins, n_frame_size)

        mag_out = self.decoder(lstm_out_m, em1, em2, em3, em4, em5, True)
        phase_out = self.decoder(lstm_out_p, ep1, ep2, ep3, ep4, ep5, False)
        
        mag_out = Padding.unpad(mag_out, pad)
        phase_out = Padding.unpad(phase_out, pad)
        
        return mag_out, phase_out

class MSELoss:
    def __init__(self):
        return
    
    def __call__(self, clean_spec_mag_pred, clean_spec_phase_pred, clean_spec_mag, clean_spec_phase):
        '''
        MSE loss
        Input - predicted and clean magnitude and phase spectrum
        Output - MSE loss calculated on magnitude and phase spectrum
        '''
        return (F.mse_loss(clean_spec_mag, clean_spec_mag_pred) + F.mse_loss(clean_spec_phase_pred, clean_spec_phase))/2
        

def build(args):
    '''
    Main function to build model based on params passed.
    '''
    # Denoise only the magnitude spectrum
    print('Model to denoise both the magnitude and phase spectrum....')

    model = ConvRecNet_Complex(args)
    print('%s pooling ...\n'%(args['pooling']))
    if args['pooling'] == 'wavelet':
        attach_hooks(model) # attach hook

    criterion = MSELoss()
        
    return model, criterion
    