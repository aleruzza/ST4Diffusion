import os
import copy
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from params import params
from loader import TextImageDataset, PretrainDataset
from create_model import create_nnmodel
from torch.utils.tensorboard import SummaryWriter
from ddpm import DDPM, EMA


def init_ddpm(params):
    
    # general parameters for the name and logger
    run_name= params['name'] # the unique name of each experiment
    logger = SummaryWriter(os.path.join("runs", run_name)) # To log
    
    # parameter for DDPM
    nT = params['nT'] # 1000, 500; DDPM time steps
    ws_test = params['ws'] #[0,0.5,2] strength of generative guidance

    # parameters for training unet
    device = params['device'] # using gpu or optionally "cpu"
    n_epoch = params['nepochs'] # 120
    lrate = params['lr']
    save_model = True
    save_dir = params['savedir']
    save_freq = params['savefreq'] #10 # the period of saving model
    ema= params['ema'] # whether to use ema
    ema_rate= params['ema_rate']
    cond = params['cond'] # if training using the conditional information
    lr_decay = params['lr_decay'] # if using the learning rate decay
    resume = params['resume'] # if resume from the trained checkpoints
    
    # parameters for sampling
    sample_freq = params['sample_freq'] # the period of sampling
    test_paradf = pd.read_csv(f'data/testpara.csv', index_col=0).loc[0:10]
    n_sample = len(test_paradf)
    test_param = torch.tensor(np.float32(np.log10(np.array(test_paradf[['PlanetMass', 'AspectRatio', 'Alpha', 'InvStokes1']]))))
    test_param =  test_param.to(device)
    
    # parameters for dataset
    batch_size = params['batch_size'] # 16
    image_size= params['image_size'] # 64
    drop_prob = params['drop_prob'] # the probability to drop the parameters for unconditional training in classifier free guidance.
    n_param = params['n_param'] # dimension of parameters
    data_dir = params['datadir'] # data directory
    pretrain = params['pretrain']
        
    # initialize the DDPM
    
    
def train(params, ddpm):
    
    # initialize the dataset
    #if pretrain load the pretraining dataset
    if params['pretrain']:
        dataset = PretrainDataset(
            folder=params['datadir'],
            image_size=128,
            shuffle=True,
            n_param=params['n_param']
        )
    else:
        dataset = TextImageDataset(
                folder=params['datadir'],
                image_size=params['image_size'],
                uncond_p=params['drop_prob'], # only used when drop_para=True
                shuffle=True,
                n_param=params['n_param'],
                drop_para=True if params['cond']==True else False
            )

    # data loader setup
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    ddpm.train_eor(params, dataloader)
    
if __name__ == "__main__":
    
    if os.path.exists(params['savedir']):
        if params['Override']:
            print('Saving directory exists, overriding old data as instructed.')
        else:
            print('WARNING! -> saving directory already exists, please run with Override=True')
            exit()
    else:
        os.mkdir(params['savedir'])

    if os.path.exists('parahist.csv'):
        oldpara = pd.read_csv('parahist.csv', index_col=0)
        params['index'] = oldpara.index[-1]+1
        newparafile = pd.concat([oldpara, pd.DataFrame([params]).set_index('index')])
    else:
        params['index'] = 0
        newparafile = pd.DataFrame([params]).set_index('index')
    newparafile.to_csv('parahist.csv')
    
    if params['resume']:
        ddpm = 0
    else:
        ddpm = DDPM(betas=(1e-4, 0.02), nT=params['nT'], device=params['device'], drop_prob=params['drop_prob'],cond=params['cond'])

    train(params=params, ddpm=ddpm)