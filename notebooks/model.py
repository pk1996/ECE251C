import torch
import torch.nn as nn
import torch.nn.functional as F


class Padding():
    @staticmethod
    def pad(x, stride):
        h, w = x.shape[-2:]

        if h % stride > 0:
            new_h = h + stride - h % stride
        else:
            new_h = h
        if w % stride > 0:
            new_w = w + stride - w % stride
        else:
            new_w = w
        lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
        lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
        pads = (lw, uw, lh, uh)

        # zero-padding by default.
        # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
        out = F.pad(x, pads, "reflect")

        return out, pads
    
    @staticmethod
    def unpad(x, pad):
        if pad[2]+pad[3] > 0:
            x = x[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            x = x[:,:,:,pad[0]:-pad[1]]
        return x

class WaveletPooling(nn.Module):
    def __init__(self, wavelet):
        super(WaveletPooling,self).__init__()
        self.upsample_ = nn.Upsample(scale_factor=2, mode='nearest')
        self.wavelet = wavelet
    
    def forward(self, x):
        bs = x.size()[0]
        FORWARD_OUTPUT_ = []
        
        # loop over input as batching not supported
        for k in range(bs):
            # coeffiecients - cx1xhxw
            coefficients = ptwt.wavedec2(x[k,:,:,:], pywt.Wavelet(self.wavelet),
                                        level=2, mode="constant")
            # 2nd order DWT
            forward_output_ = ptwt.waverec2([coefficients[0], coefficients[1]], pywt.Wavelet(self.wavelet))
            
            # permute dim - 1xcxhxw
            FORWARD_OUTPUT_.append(torch.permute(forward_output_, [1,0,2,3]))
        
        FORWARD_OUTPUT_ = torch.cat(FORWARD_OUTPUT_, dim = 0)
        
        if x.shape[-1]/2 != FORWARD_OUTPUT_.shape[-1]:
            FORWARD_OUTPUT_ = FORWARD_OUTPUT_[:,:,:,:]
        if x.shape[-2]/2 != FORWARD_OUTPUT_.shape[-2]:
            FORWARD_OUTPUT_ = FORWARD_OUTPUT_[:,:,:-1,:]
        
        return FORWARD_OUTPUT_
    
def wavelet_pooling_hook(module, inp, out):
    '''
    inp - gradient output from the layer
    out - gradient inp to layer 
    '''    
    # Computing gradient using paper.
    bs = out[0].size()[0]
    BACKWARD_OUTPUT_ = []

    # loop over input as batching not supported
    for k in range(bs):
        ## 1. 1st order DWT
        coefficients = ptwt.wavedec2(torch.squeeze(out[0][k]), pywt.Wavelet("haar"),
                                        level=1)#, mode="constant")
        ## 2. upsample subbands
        # LL
        upsampled_subbands_ = coefficients
        
        # LH, HL, HH
        upsampled_subbands_.append([])
        for k in range(len(coefficients[1])):
            upsampled_subbands_[-1].append(module.upsample_(coefficients[1][k]))
        upsampled_subbands_[-1] = tuple(upsampled_subbands_[-1])  

        ## 3. IDWT
        backward_output_ = ptwt.waverec2(upsampled_subbands_, pywt.Wavelet("haar"))
        BACKWARD_OUTPUT_.append(backward_output_.permute(1,0,2,3))
    
    BACKWARD_OUTPUT_ = torch.cat(BACKWARD_OUTPUT_, dim = 0)
    
    cw = 4-inp[0].shape[2]%4
    ch = 4-inp[0].shape[3]%4
    
    if cw != 4:
        BACKWARD_OUTPUT_ = BACKWARD_OUTPUT_[:,:,:-cw,:]
    
    if ch != 4:
        BACKWARD_OUTPUT_ = BACKWARD_OUTPUT_[:,:,:,:-ch]
        
    BACKWARD_OUTPUT_SHAPE_ = BACKWARD_OUTPUT_.shape
    
    return [BACKWARD_OUTPUT_]

class Pooling(nn.Module):
    def __init__(self, params):
        super().__init__()
        if params['pooling'] == 'wavelet':
            print('Waveler pooling ....')
            self.pool = WaveletPooling(params['wavelet'])
        else:
            print('Max pooling ....')
            self.pool = nn.MaxPool2d(3,2,1)
    
    def forward(self, x):
        return self.pool(x)
    
class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, params):
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),            
            padding=(1, 1)
        )
        self.pool = Pooling(params)
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [batch_size, num_channels, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv_layer(x)
        x = self.pool(x)
        x = self.norm(x)
        x = self.activation(x)
        
        return x
    
    
class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False):
        super().__init__()
        
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding = (1,1),
            output_padding = (1,1)
        )
        
        
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        
        return x
    
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
        self.lstm_layer = nn.LSTM(input_size=256*9, hidden_size=256*9, num_layers=2, batch_first=True)
        
        self.tran_conv_block_1 = CausalTransConvBlock(256 + 256, 128)
        self.tran_conv_block_2 = CausalTransConvBlock(128 + 128, 64)
        self.tran_conv_block_3 = CausalTransConvBlock(64 + 64, 32)
        self.tran_conv_block_4 = CausalTransConvBlock(32 + 32, 16)
        self.tran_conv_block_5 = CausalTransConvBlock(16 + 16, 1, is_last=True)

        
    def forward(self, x):
        x, pad = Padding.pad(x, 32)
        
        self.lstm_layer.flatten_parameters()
        e1 = self.conv_block_1(x)
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
        return d5

def build(args):
    '''
    Main function to build model based on params passed.
    '''
    