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


# Train function ----------------
def train_one_epoch(model, train_dataloader, criterion, optimizer, args):
    '''
    Training loop 
    '''
    num_batches = len(train_dataloader)
    LOSS = []
    model.train()     # switch to train mode
    for itr in tqdm(range(num_batches)):
        clean_spec, noisy_spec = next(iter(train_dataloader))
        
        clean_spec = clean_spec[0].to(args['device'])
        noisy_spec = noisy_spec[0].to(args['device'])
        
        # compute the magnitude response of clean and noisy spectrograms
        clean_spec_mag, clean_spec_phase = get_mag_phase(clean_spec)
        noisy_spec_mag, noisy_spec_phase = get_mag_phase(noisy_spec)
                
        # Forward
        clean_spec_mag_pred, clean_spec_phase_pred = model(noisy_spec_mag, noisy_spec_phase)
        loss = criterion(clean_spec_mag_pred, clean_spec_phase_pred, clean_spec_mag, clean_spec_phase)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        LOSS.append(loss.item())
    
    return LOSS

# --------------------------------
# Main 
# --------------------------------

def main(config_file = 'ml_haar.yaml'):
    # Settings --------------

    # Get configs
    args = load_config(config_file)
    print(args)
    print("")

    NUM_EPOCHS = args['num_epochs']
    TRAIN_BATCH_SIZE = args['batch_size']
    EVAL_BATCH_SIZE = 1
    EVAL_EVERY = args['eval_every']
    LR = args['learning_rate']
    LR_STEP_SIZE = ['lr_step_size']   # How often to decrease learning by gamma.
    GAMMA = ['gamma']         # LR is multiplied by gamma on schedule

    # Set up experiment folder
    BASE_DIR = Path(os.getcwd()).parent
    EXP_PATH = os.path.join(BASE_DIR, 'experiments', args['name'])
    os.makedirs(EXP_PATH, exist_ok = True)

    loss_f = open(os.path.join(EXP_PATH, 'loss.pkl'), 'wb')
    eval_f = open(os.path.join(EXP_PATH, 'eval.pkl'), 'wb')

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
    dataset_train = build_dataset('train', samples = 2000)
    dataloader_train = DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, collate_fn=custom_collate_fn)

    # eval dataset
    dataset_eval = build_dataset('val', samples = 500)
    dataloader_eval = DataLoader(dataset_eval, batch_size=EVAL_BATCH_SIZE, collate_fn=custom_collate_fn)


    # Build model & criterion --------------
    model, criterion = build_model(args['model'])
    model.to(device)

    # if ckpt is not None:
        # Load checkpoint
        # TODO     

    # Setup --------------

    # TODO - add support for learning rate scheduler
    optimizer = optim.SGD(model.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=GAMMA)


    # Training --------------

    # Initialize
    LOSS_DICT = {}
    EVAL_DICT = {}
    print('Starting training ....')


    for epoch in range(NUM_EPOCHS):
        print('Train %d/%d'%(epoch, NUM_EPOCHS))
        loss = train_one_epoch(model, dataloader_train, criterion, optimizer, args)
        LOSS_DICT[epoch] = loss

    #     # step optimizer
    #     scheduler.step()

        # evaluate periodically
        if (epoch+1) % EVAL_EVERY == 0:
            eval_res = evaluate(model, dataloader_eval, criterion, args)
            EVAL_DICT[epoch] = eval_res

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(EXP_PATH, 'checkpoint.pth'))

    # Save loss and eval results.
    pickle.dump(LOSS_DICT, loss_f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(EVAL_DICT, eval_f, protocol=pickle.HIGHEST_PROTOCOL)
                        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="specify config file")
    parser.add_argument("--config", type=str, default = '../configs/m1_pool.yaml', help="specify config file")
    args = parser.parse_args()
    main(args.config)




