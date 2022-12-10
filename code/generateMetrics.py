"""
To run trained models on test set to generate
stats - MSE, SNR, PESQ, STOI
"""

import os
from tqdm import tqdm
from pathlib import Path
import pickle
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import build_model
from data import build_dataset
from evaluate import evaluate
from utils import *


# --------------------------------
# Main 
# --------------------------------

def main(config_file = 'local_exp.yaml'):
    # Settings --------------

    # Get configs
    args = load_config(config_file)
    print(args)
    print("")

    # Set up experiment folder
    BASE_DIR = Path(os.getcwd()).parent
    EXP_PATH = os.path.join(BASE_DIR, 'experiments', args['name'])
    os.makedirs(EXP_PATH, exist_ok = True)

    # define device type - cuda:0 or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args['device'] = device

    # Additional Info when using cuda
    if device.type == 'cuda':
        print("Number of GPU devices:", torch.cuda.device_count())
        print("GPU device name:", torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 3), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 3), 'GB \n')

    # Build dataloader --------------
    # train dataset
    dataset_test = build_dataset('test', samples = 100)
    dataloader_test = DataLoader(dataset_test, batch_size=1, collate_fn=custom_collate_fn)

    # Build model & criterion --------------
    model, criterion = build_model(args['model'])
    model.to(device)
    
    # Create pkl file to save res
    EVAL_RES_DIR = os.path.join(BASE_DIR, 'experiments', 'evaluate')
    os.makedirs(EVAL_RES_DIR, exist_ok = True)
    eval_f = open(os.path.join(EVAL_RES_DIR, 'eval_%s.pkl'%(args['name'])), 'wb')

    # Load checkpoint
    ckpt = torch.load(os.path.join(EXP_PATH, 'checkpoint.pth'))
    model.load_state_dict(ckpt['state_dict'])

    print('Starting evaluating ....')
    
    eval_res = evaluate(model, dataloader_test, criterion, args)
    # Compute mean stats
    print("Test Results on %s"%(args['name']))
    for k,v in eval_res.items():
        eval_res[k] = sum(v)/len(v)
        print(k + ": " + str(eval_res[k]))
        
        
    # Save loss and eval results.
    pickle.dump(eval_res, eval_f, protocol=pickle.HIGHEST_PROTOCOL)
                        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="specify config file")
    parser.add_argument("--config", type=str, default = '../configs/local_exp.yaml', help="specify config file")
    args = parser.parse_args()
    main(args.config)




