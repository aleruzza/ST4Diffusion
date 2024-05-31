name = 'testname'
################### Normalization functions ###################################
def scaleandlog(data, scale):
    data = np.nan_to_num(data)
    return np.log10(1 + data/scale)

def nonorm(data, scale):
    return data/scale
params = {
    'name': name,
    'nT': 5,
    'ws': [0, 0.5], 
    'device': 'cpu', 
    'nepochs': 10,
    'lr': 1e-4,
    'savedir': f'./outputs/{name}',
    'datadir': f'./data/',
    'Override': True,
    'savefreq': 16,
    'ema': True,
    'ema_rate': 0.995, 
    'cond': True,
    'lr_decay': False,
    'resume': False,
    'sample_freq': 10, 
    'batch_size': 64,
    'logima_freq': 10,
    'norm': scaleandlog,
    'n_test_log_images': 50,
    'image_size': 128,
    'drop_prob': 0.25,
    'n_param' : 6,
    'pretrain': True,
    'n_pretrain': 10000 #note: it must be <101,000
}