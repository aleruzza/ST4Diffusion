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

    
    
def train(params, ddpm):
    
    # initialize the dataset
    #if pretrain load the pretraining dataset
    if params['pretrain']:
        dataset = PretrainDataset(
            folder=params['datadir'],
            image_size=128,
            shuffle=True,
            n_param=params['n_param'],
            n_pretrain = params['n_pretrain']
        )
        test_param = None
    else:
        dataset = TextImageDataset(
                folder=params['datadir'],
                image_size=params['image_size'],
                uncond_p=params['drop_prob'], # only used when drop_para=True
                shuffle=True,
                n_param=params['n_param'],
                drop_para=True if params['cond']==True else False
            )
        test_paradf = pd.read_csv(f'data/testpara.csv', index_col=0).loc[0:10]
        test_param = torch.tensor(np.float32(np.log10(np.array(test_paradf[['PlanetMass', 'AspectRatio', 'Alpha', 'InvStokes1', 'SigmaSlope', 'FlaringIndex']]))))
        test_param =  test_param.to(params['device'])

    # data loader setup
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    ddpm.train_eor(params, dataloader, test_param=test_param)
    
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
        ddpm = DDPM(betas=(1e-4, 0.02), nT=params['nT'],
                    n_param=params['n_param'], device=params['device'], drop_prob=params['drop_prob'],cond=params['cond'])

    train(params=params, ddpm=ddpm)