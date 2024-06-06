import torch
import torch.nn as nn
import numpy as np
import os
import copy
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import gc
import wandb

from create_model import create_nnmodel
from torch.utils.tensorboard import SummaryWriter
from create_model import create_nnmodel

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model):
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

def ddpm_schedules(beta1, beta2, T, device):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = torch.linspace(beta1, beta2, T, dtype=torch.float32).to(device)
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    alphabar_t = torch.cumprod(alpha_t, dim=0)

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, betas, nT, device, drop_prob=0.1, cond=False, n_param=6, image_size=128, ema=True, ema_rate=0.995):
        super(DDPM, self).__init__()

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], nT,device).items():
            self.register_buffer(k, v)

        self.nT = nT
        self.device = device
        self.drop_prob = drop_prob
        self.cond = cond
        self.n_param = n_param
        self.image_size= image_size
        self.ema_flag = ema
        self.ema_rate = ema_rate
        
        # initialize the unet
        self.nn_model=create_nnmodel(n_param=self.n_param,image_size=image_size)
        self.nn_model.train()
        self.nn_model.to(device)
        
        if self.ema_flag:
            self.ema = EMA(ema_rate)
            self.ema_model = copy.deepcopy(self.nn_model).eval().requires_grad_(False)


    def noised(self, x):
        """
        this method is used in denoising process, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.nT+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, nT)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts-1, None,None,None] * x
            + self.sqrtmab[_ts-1, None,None,None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. 

        return noise, x_t, _ts


    @torch.inference_mode()
    def sample(self, nn_model, n_sample, size, test_param,  device =None,  guide_w = 0.0, store=False):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with parameter (tokens) and the other with empty (0) tokens.
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance
        nn_model.eval()
        
        if device is None:
            device=self.device
    
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        if self.cond == True:
            c_i = test_param
            #uncond_tokens = torch.tensor(np.float32(np.zeros(test_param.shape[-1]))).to(device)
            #uncond_tokens=uncond_tokens.repeat(int(n_sample),1)
            #c_i = torch.cat((c_i, uncond_tokens), 0)

        x_i_store = [] # keep track of generated steps in case want to plot something
        
        zz = torch.randn(self.nT-1, n_sample, *size).to(device)
        for i in tqdm(range(self.nT, 0, -1)):
            t_is = torch.tensor([i]).to(device)
            t_is = t_is.repeat(n_sample)
            
            z = zz[i-2] if i>1 else 0
            #z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # double batch
            if self.cond == True:
                #x_i = x_i.repeat(2,1,1,1)
                #t_is = t_is.repeat(2)

                # split predictions and compute weighting
                eps = nn_model(x_i, t_is,c_i)
                #eps1 = eps[:n_sample]
                #eps2 = eps[n_sample:]
                #eps = (1+guide_w)*eps1 - guide_w*eps2
                
                #x_i = x_i[:n_sample]
                x_i = (
                    self.oneover_sqrta[i-1] * (x_i - eps * self.mab_over_sqrtmab[i-1])
                    + self.sqrt_beta_t[i-1] * z
                ) 
            else:
                eps = nn_model(x_i, t_is)
                x_i = (
                    self.oneover_sqrta[i-1] * (x_i - eps * self.mab_over_sqrtmab[i-1])
                    + self.sqrt_beta_t[i-1] * z
                )
             
            # store only part of the intermediate steps
            #if i%50==0 or i==self.nT or i<8:
             #   x_i_store.append(x_i.detach().cpu().numpy())
        if store:
            x_i_store = np.array(x_i_store)
        return x_i, x_i_store

    def getmse(im1, im2, x, y):
        xx, yy = np.meshgrid(x,y)
        rr = np.sqrt(xx**2+yy**2)
        return (((im1-im2)**2)*((rr<3) & (rr>0.3))).mean()

    def train_eor(self, params, dataloader, test_param, x_test):
    
        # general parameters for the name and logger
        run_name= params['name'] # the unique name of each experiment
        ws_test = params['ws'] #[0,0.5,2] strength of generative guidance

        # parameters for training unet
        device = params['device'] # using gpu or optionally "cpu"
        n_epoch = params['nepochs'] # 120
        lrate = params['lr']
        save_model = True
        save_dir = params['savedir']
        save_freq = params['savefreq'] #10 # the period of saving model
    
        cond = params['cond'] # if training using the conditional information
        lr_decay = params['lr_decay'] # if using the learning rate decay

        # parameters for sampling
        sample_freq = params['sample_freq'] # the period of sampling
        if test_param is None:
            n_sample=0
        else:
            n_sample = len(test_param)
        test_param = test_param
        
        # parameters for dataset
        image_size= params['image_size'] # 64
        
        ########################
        # parameters to be optimized
        params_to_optimize = [
            {'params': self.nn_model.parameters()}
        ]

        # number of parameters to be trained
        number_of_params = sum(x.numel() for x in self.nn_model.parameters())
        print(f"Number of parameters for unet: {number_of_params}")

        # define the loss function
        loss_mse = nn.MSELoss()

        length = len(dataloader)

        # initialize optimizer
        optim = torch.optim.Adam(params_to_optimize, lr=lrate)

        # whether to use ema
        

        ###################      
        ## training loop ##
        ###################
        
        wandb.watch(self.nn_model, criterion=loss_mse, log='all', log_freq=10)
        for ep in range(n_epoch):
            print(f'epoch {ep}')
            self.train() 
            self.nn_model.train()
            if self.ema_flag:
                self.ema_model.train()#this only sets the module in Training mode
            # linear lrate decay
            if lr_decay:
                optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

            # data loader with progress bar
            pbar = tqdm(dataloader)
            
            mse_list = np.array([])
            for i,(x, c) in enumerate(pbar):
                optim.zero_grad() #resets the gradients
                x = x.to(device) 
                noise,xt,ts = self.noised(x)
                if cond == True:
                    c = c.to(device)
                    noise_pred = self.nn_model(xt, ts, c)
                else:
                    noise_pred = self.nn_model(xt, ts)
                loss=loss_mse(noise, noise_pred)
                loss.backward() #computes the gradients wrt every parameter

                pbar.set_description(f"loss: {loss.item():.4f}")
                optim.step()

                # ema update
                if self.ema_flag:
                    self.ema.step_ema(self.ema_model, self.nn_model)

                # logging loss
                mse_list = np.append(mse_list, [loss.item()])
                #logger.add_scalar("MSE", loss.item(), global_step=ep * length + i)

                
            wandb.log({'loss': mse_list.mean(), 'epoch': ep})
            # save model
            if save_model:
                if ep%save_freq==0:
                    torch.save(self.state_dict(), save_dir + f"model__epoch_{ep}_test_{run_name}.pth")
                    print('saved model at ' + save_dir + f"model__epoch_{ep}_test_{run_name}.pth")
                        
            # sample the image
            if n_sample>0 and ep%sample_freq==0:
                self.nn_model.eval()
                with torch.no_grad():
                        
                    #x_gen_tot_ema=[]
                    #x_gen_tot = []

                    # only output the image x0, omit the stored intermediate steps, OTHERWISE, uncomment 
                    # line 142, 143 and output 'x_gen, x_store = ' here.
                    x_gen, _ = self.sample(self.nn_model,n_sample, (1,image_size,image_size), device=device, test_param=test_param)
                    x_gen_ema, _ = self.sample(self.ema_model,n_sample, (1,image_size,image_size), device=device, test_param=test_param)
                    
                    x_gen = x_gen.to('cpu')
                    x_gen_ema = x_gen_ema.to('cpu')
                    mse_test = ((x_gen-x_test)**2).mean()
                    mse_test_ema = ((x_gen_ema-x_test)**2).mean()
                    
                    wandb.log({'mse_test': mse_test, 'mse_test_ema': mse_test_ema, 'epoch': ep})
                    
                    #log some test images
                    if ep%params['logima_freq']==0:
                        images = []
                        for i in range(params['n_test_log_images']):
                            image = wandb.Image(torch.tensor(np.float32(x_gen[i])), mode='F')
                            images.append(image)
                        wandb.log({"testset_emulations": images})
                    if ep%params['logima_freq']==0:
                        images = []
                        for i in range(params['n_test_log_images']):
                            image = wandb.Image(torch.tensor(np.float32(x_gen_ema[i])), mode='F')
                            images.append(image)
                        wandb.log({"testset_emulations_ema": images})
                    if ep==0:
                        images = []
                        for i in range(params['n_test_log_images']):
                            image = wandb.Image(torch.tensor(np.float32(x_test[i])), mode='F')
                            images.append(image)
                        wandb.log({"testset_simulations": images})
                    
                    del x_pred_t
                    torch.cuda.empty_cache()
                    
                    #### ---------------------------> Code below is for saving the images <--------------------------------- ####
                    #x_gen_tot.append(np.array(x_gen.cpu()))
                    #x_gen_tot=np.array(x_gen_tot)
                    #x_gen_tot_ema.append(np.array(x_gen_ema.cpu()))
                    #x_gen_tot_ema=np.array(x_gen_tot_ema)

                    #sample_save_path_final = os.path.join(save_dir, f"train-{ep}xscale_{w}_test_{run_name}.npy")
                    #np.save(str(sample_save_path_final),x_gen_tot)
                    #sample_save_path_final = os.path.join(save_dir, f"train-{ep}xscale_{w}_test_{run_name}_ema.npy")
                    #np.save(str(sample_save_path_final),x_gen_tot_ema)
                    