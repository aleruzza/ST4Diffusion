import torch
import torch.nn as nn
import numpy as np

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
    def __init__(self, betas, n_T, device, drop_prob=0.1, cond=False):
        super(DDPM, self).__init__()

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T,device).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.cond = cond

    def noised(self, x):
        """
        this method is used in denoising process, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts-1, None,None,None] * x
            + self.sqrtmab[_ts-1, None,None,None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. 

        return noise, x_t, _ts

    def sample(self, nn_model, n_sample, size, device, test_param, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with parameter (tokens) and the other with empty (0) tokens.
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        if self.cond == True:

            c_i = test_param

            uncond_tokens = torch.tensor(np.float32(np.zeros(test_param.shape[-1]))).to(device)
            uncond_tokens=uncond_tokens.repeat(int(n_sample),1)

            c_i = torch.cat((c_i, uncond_tokens), 0)

        x_i_store = [] # keep track of generated steps in case want to plot something
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i]).to(device)
            t_is = t_is.repeat(n_sample)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # double batch
            if self.cond == True:
                x_i = x_i.repeat(2,1,1,1)
                t_is = t_is.repeat(2)

                # split predictions and compute weighting
                eps = nn_model(x_i, t_is,c_i)
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1+guide_w)*eps1 - guide_w*eps2
                x_i = x_i[:n_sample]
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
            #if i%20==0 or i==self.n_T or i<8:
            #    x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store
