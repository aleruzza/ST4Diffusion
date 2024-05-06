import time
from pathlib import Path
from random import randint, random
import torch as th
import pandas as pd
from torch.utils.data import Dataset
import numpy as np


def get_image_files_narray(base_path):
    image_files = np.load(f'{base_path}/data.npy')
    return image_files

def get_labels_narray(base_path):
    paradf = pd.read_csv(f'{base_path}/run4.csv', index_col=0)
    labels = np.array(paradf[['PlanetMass', 'AspectRatio', 'Alpha', 'InvStokes1']])
    labels = np.log10(labels)
    print('I am fetching the parameters')
    return labels

def get_pretraining_data(base_path, n=10):
    dataset = np.load(f'{base_path}/swe_data.npy')
    dataset = dataset.reshape(-1, 128, 128)
    np.random.shuffle(dataset)
    return dataset[0:n]




class PretrainDataset(Dataset):
    """Dataset for pretraining

 
    """
 
    def __init__(
        self,
        folder="",
        image_size=128,
        shuffle=False,
        n_param =6,
    ):
        """Init

        Parameters
        ----------
        folder : str, optional
            folder where the .npy files are stored, by default ""
        image_size : int, optional
            pixel dimension of the images in the dataset, by default 128
        shuffle : bool, optional
            enables shuffling of the dataset, by default False
        n_param : int, optional
            number of conditional parameters for which the model is built.
            Note that during pretraining all these parameters
            will be set to 0, by default 6
        """        
        super().__init__()
        folder = Path(folder)
        self.data = get_pretraining_data(folder)
        self.shuffle = shuffle
        self.prefix = folder
        self.image_size = image_size
        self.n_param = n_param

    def __len__(self):
        return len(self.data)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        tokens = np.float32(np.array([0]).repeat(self.n_param)) #array([0,0]) for uncondional training
        original_image = np.float32(self.data[ind])
        arr = np.expand_dims(original_image,axis=0) # only one channel

        return th.tensor(arr),th.tensor(np.float32(tokens))

    

class TextImageDataset(Dataset):
    def __init__(
        self,
        folder="",
        image_size=64,
        shuffle=False,
        uncond_p=0.0,
        n_param=2,
        drop_para=False
    ):
        super().__init__()
        folder = Path(folder)

        self.data = get_image_files_narray(folder)
        self.labels = get_labels_narray(folder)

        self.n_param = n_param

        self.shuffle = shuffle
        self.prefix = folder
        self.image_size = image_size
        self.uncond_p = uncond_p
        self.drop_para=drop_para

    def __len__(self):
        return len(self.labels)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def get_para(self, ind):
        return self.labels[ind]

    def __getitem__(self, ind):
        
        if random() < self.uncond_p and self.drop_para == True:
            tokens = np.float32(np.array([0]).repeat(self.n_param)) #array([0,0]) for uncondional training
        else:
            tokens = self.get_para(ind)

        original_image = np.float32(self.data[ind])

        arr = np.expand_dims(original_image,axis=0) # only one channel
        arr = arr*2 - 1 # image array already normalized to [0,1], here further to [-1,1]
        return th.tensor(arr),th.tensor(np.float32(tokens))

