from torch.utils.data import Dataset
from .utils import load_pkl_data

# Dataset ---------------------
class SpeechDataset(Dataset):
    def __init__(self, split='train', num_samples=1000, alpha=0.8):
        assert split in ['train', 'val', 'test'], "Invalid split"
        self.split = split
        self.alpha = alpha
        self.num_samples = num_samples
        self.clean_mixed_data_dict = load_pkl_data(split, num_samples)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        stft_clean = self.clean_mixed_data_dict['clean'][idx]
        stft_mixed = self.clean_mixed_data_dict['mixed'][idx]
        
        return stft_clean, stft_mixed
    

def build(split = 'train', samples = 2000):
    dataset = SpeechDataset(split = split, num_samples = samples)
    return dataset