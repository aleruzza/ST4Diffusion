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



def train_eor():
    
    # general parameters for the name and logger
    run_name= params['name'] # the unique name of each experiment
    logger = SummaryWriter(os.path.join("runs", run_name)) # To log
    
    # parameter for DDPM
    n_T = params['nT'] # 1000, 500; DDPM time steps
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
    
    ########################
    ## ready for training ##
    ########################
    # initialize the DDPM
    ddpm = DDPM(betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=drop_prob,cond=cond)

    # initialize the unet
    nn_model=create_nnmodel(n_param=n_param,image_size=image_size)
    nn_model.train()
    nn_model.to(device)

    # parameters to be optimized
    params_to_optimize = [
        {'params': nn_model.parameters()}
    ]

    # number of parameters to be trained
    number_of_params = sum(x.numel() for x in nn_model.parameters())
    print(f"Number of parameters for unet: {number_of_params}")

    # optionally load a model
    if resume:
        ddpm.load_state_dict(torch.load(os.path.join(save_dir, f"train-{ep}xscale_test_{run_name}.npy")))

    # define the loss function
    loss_mse = nn.MSELoss()

    # initialize the dataset
    #if pretrain load the pretraining dataset
    if pretrain:
        dataset = PretrainDataset(
            folder=data_dir,
            image_size=128,
            shuffle=True,
            n_param=6
        )
    else:
        dataset = TextImageDataset(
                folder=data_dir,
                image_size=image_size,
                uncond_p=drop_prob, # only used when drop_para=True
                shuffle=True,
                n_param=n_param,
                drop_para=True if cond==True else False
            )

    # data loader setup
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == device),
    )
    length = len(dataloader)

    
    # initialize optimizer
    optim = torch.optim.Adam(params_to_optimize, lr=lrate)

    # whether to use ema
    if ema:
        ema = EMA(ema_rate)
        if resume:
            ema_model = DDPM(nn_model=nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=drop_prob,cond=cond)
            ema_model.load_state_dict(torch.load(os.path.join(save_dir, f"train-{ep}xscale_test_{run_name}_ema.npy")))
        else:
            ema_model = copy.deepcopy(nn_model).eval().requires_grad_(False)

    ###################      
    ## training loop ##
    ###################
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()  #this only sets the module in Training mode
        # linear lrate decay
        if lr_decay:
            optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        # data loader with progress bar
        pbar = tqdm(dataloader)
        for i,(x, c) in enumerate(pbar):
            optim.zero_grad() #resets the gradients
            x = x.to(device) 
            noise,xt,ts = ddpm.noised(x)
            if cond == True:
                c = c.to(device)
                noise_pred = nn_model(xt, ts, c)
            else:
                noise_pred = nn_model(xt, ts)
            loss=loss_mse(noise, noise_pred)
            loss.backward() #computes the gradients wrt every parameter

            pbar.set_description(f"loss: {loss.item():.4f}")
            optim.step()

            # ema update
            if ema:
                ema.step_ema(ema_model, nn_model)

            # logging loss
            logger.add_scalar("MSE", loss.item(), global_step=ep * length + i)

            
            # save model
            if save_model:
                model_state = {
                    'epoch': ep,
                    'unet_state_dict': nn_model.state_dict(),
                    'ema_unet_state_dict': ema_model.state_dict()
                    }
                torch.save(model_state, save_dir + f"model_epoch_{ep}_test_{run_name}.tar")
                print('saved model at ' + save_dir + f"model__epoch_{ep}_test_{run_name}.pth")
                
            # sample the image
            if ep%sample_freq==0:
                nn_model.eval()
                with torch.no_grad():

                    # loop over the guidance scale
                    for w in ws_test: 
                        
                        x_gen_tot_ema=[]
                        x_gen_tot = []

                        # only output the image x0, omit the stored intermediate steps, OTHERWISE, uncomment 
                        # line 142, 143 and output 'x_gen, x_store = ' here.
                        x_gen, _ = ddpm.sample(nn_model,n_sample, (1,image_size,image_size), device, test_param=test_param, guide_w=w)
                        x_gen_ema, _ = ddpm.sample(ema_model,n_sample, (1,image_size,image_size), device, test_param=test_param, guide_w=w)

                        x_gen_tot.append(np.array(x_gen.cpu()))
                        x_gen_tot=np.array(x_gen_tot)
                        x_gen_tot_ema.append(np.array(x_gen_ema.cpu()))
                        x_gen_tot_ema=np.array(x_gen_tot_ema)

                        sample_save_path_final = os.path.join(save_dir, f"train-{ep}xscale_{w}_test_{run_name}.npy")
                        np.save(str(sample_save_path_final),x_gen_tot)
                        sample_save_path_final = os.path.join(save_dir, f"train-{ep}xscale_{w}_test_{run_name}_ema.npy")
                        np.save(str(sample_save_path_final),x_gen_tot_ema)

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
    train_eor()