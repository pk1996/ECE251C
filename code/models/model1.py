"""
Implements model1 which only cleans the magnitude spectrum
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_util import *

class ConvRecNet(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output:[batch size, channels=1, T, n_fft]
    """
    
    def __init__(self, params):
        super(ConvRecNet, self).__init__()
        
        # Encoder
        self.conv_block_1 = CausalConvBlock(1, 16, params)
        self.conv_block_2 = CausalConvBlock(16, 32, params)
        self.conv_block_3 = CausalConvBlock(32, 64, params)
        self.conv_block_4 = CausalConvBlock(64, 128, params)
        self.conv_block_5 = CausalConvBlock(128, 256, params)
        
        # LSTM
        x = 17 # 9
        self.lstm_layer = nn.LSTM(input_size=256*x, hidden_size=256*x, num_layers=2, batch_first=True)
        
        self.tran_conv_block_1 = CausalTransConvBlock(256 + 256, 128)
        self.tran_conv_block_2 = CausalTransConvBlock(128 + 128, 64)
        self.tran_conv_block_3 = CausalTransConvBlock(64 + 64, 32)
        self.tran_conv_block_4 = CausalTransConvBlock(32 + 32, 16)
        self.tran_conv_block_5 = CausalTransConvBlock(16 + 16, 1, is_last=True)

        
    def forward(self, mag, phase):
        """
        Input - Noisy magnitude and phase spectrums
        Output - Cleaned up magnitude and phase spectrums
        
        Note - This model doesn't do anything with the phase spectrum
        """
        mag, pad = Padding.pad(mag, 32)
        
        self.lstm_layer.flatten_parameters()
        e1 = self.conv_block_1(mag)
        e2 = self.conv_block_2(e1)
        e3 = self.conv_block_3(e2)
        e4 = self.conv_block_4(e3)
        e5 = self.conv_block_5(e4)  # [2, 256, 4, 200]
                
        batch_size, n_channels, n_f_bins, n_frame_size = e5.shape

        # [2, 256, 4, 200] = [2, 1024, 200] => [2, 200, 1024]
        lstm_in = e5.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
        lstm_out, _ = self.lstm_layer(lstm_in)  # [2, 200, 1024]
        lstm_out = lstm_out.permute(0, 2, 1).reshape(batch_size, n_channels, n_f_bins, n_frame_size)  # [2, 256, 4, 200]
        
        d1 = self.tran_conv_block_1(torch.cat((lstm_out, e5), 1))
        d2 = self.tran_conv_block_2(torch.cat((d1, e4), 1))
        d3 = self.tran_conv_block_3(torch.cat((d2, e3), 1))
        d4 = self.tran_conv_block_4(torch.cat((d3, e2), 1))
        d5 = self.tran_conv_block_5(torch.cat((d4, e1), 1))
        
        d5 = Padding.unpad(d5, pad)
        return d5, phase

class MSELoss:
    def __init__(self):
        return
    
    def __call__(self, clean_spec_mag_pred, clean_spec_phase_pred, clean_spec_mag, clean_spec_phase):
        '''
        MSE loss
        Input - predicted and clean magnitude and phase spectrum
        Output - MSE loss calculated on magnitude and phase spectrum
        '''
        return F.mse_loss(clean_spec_mag, clean_spec_mag_pred)
        

def build(args):
    '''
    Main function to build model based on params passed.
    '''
    # Denoise only the magnitude spectrum
    print('Model to denoise only the magnitude spectrum....')

    model = ConvRecNet(args)
    print('%s pooling ...\n'%(args['pooling']))
    if args['pooling'] == 'wavelet':
        attach_hooks(model) # attach hook

    criterion = MSELoss()
        
    return model, criterion
    