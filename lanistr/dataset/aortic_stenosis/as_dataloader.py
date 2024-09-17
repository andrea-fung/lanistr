from os.path import join
from random import randint
from typing import List, Dict, Union
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip
import warnings
from random import lognormvariate
from random import seed
import torch.nn as nn
import random
from dataset.aortic_stenosis.as_utils import load_as_data, preprocess_as_data, fix_leakage

seed(42)
torch.random.manual_seed(42)
np.random.seed(42)

img_path_dataset = '/workspace/data/as_tom/annotations-all.csv'
tab_path_dataset = '/workspace/data/as_tom/finetuned_df.csv'
dataset_root = r"/workspace/data/as_tom"

# filter out pytorch user warnings for upsampling behaviour
warnings.filterwarnings("ignore", category=UserWarning)

def mat_loader(path):
    mat = loadmat(path)
    if 'cine' in mat.keys():    
        return loadmat(path)['cine']
    if 'cropped' in mat.keys():    
        return loadmat(path)['cropped']
    
def get_as_dataloader(config, split, mode):
    '''
    Uses the configuration dictionary to instantiate AS dataloaders

    Parameters
    ----------
    config : Configuration dictionary
        follows the format of get_config.py
    split : string, 'train'/'val'/'test' for which section to obtain
    mode : string, 'train'/'val'/'test' for setting augmentation/metadata ops

    Returns
    -------
    Training, validation or test dataloader with data arranged according to
    pre-determined splits

    '''
    
    if mode=='train':
        flip=config['flip_rate']
        tra = True
        bsize = config['batch_size']
    elif mode=='val':
        flip = 0.0
        tra = False
        #bsize = config['batch_size']
        bsize = 12
    elif mode=='test':
        flip = 0.0
        tra = False
        bsize = 1
        
    fr = 16
    
    # read in the data directory CSV as a pandas dataframe
    raw_dataset = pd.read_csv(img_path_dataset)
    dataset = pd.read_csv(img_path_dataset)
        
    # append dataset root to each path in the dataframe
    dataset['path'] = dataset['path'].map(lambda x: join(dataset_root, x))
    view = config['view']
        
    if view in ('plax', 'psax'):
        dataset = dataset[dataset['view'] == view]
    elif view != 'all':
        raise ValueError(f'View should be plax, psax or all, got {view}')
       
    # remove unnecessary columns in 'as_label' based on label scheme
    scheme = {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
    dataset = dataset[dataset['as_label'].isin( scheme.keys() )]

    #load tabular dataset
    tab_train, tab_val, tab_test = load_as_data(csv_path = tab_path_dataset,
                                                drop_cols = config['drop_cols'],
                                                num_ex = config['num_ex'],
                                                scale_feats = config['scale_feats'])
                                                

    #perform imputation 
    train_set, val_set, test_set, all_cols = preprocess_as_data(tab_train, tab_val, tab_test, config['categorical_cols'])
    
    # Take train/test/val
    if split in ('train', 'val', 'test', 'ulb'):
        dataset = dataset[dataset['split'] == split]
        if split=='train':
            tab_dataset = train_set
        elif split=='val':
            tab_dataset = val_set
        elif split=='test':
            tab_dataset = test_set
    elif split == 'train_all':
        dataset = dataset[dataset['split'].isin(['train','ulb'])]
        tab_dataset = train_set
    elif split != 'all':
        raise ValueError(f'View should be train/val/test/all, got {split}')
    
    #Fix data leakage 
    if split == 'train':
        dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
    elif split == 'val':
        dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
    elif split == 'test':
        dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
        
    dset = AorticStenosisDataset(img_path_dataset=dataset, 
                                tab_dataset=tab_dataset,
                                mat_loader=mat_loader,
                                split=split,
                                transform=tra,
                                normalize=True,
                                frames=fr,
                                flip_rate=flip,
                                label_scheme = scheme)
    
    if mode=='train':
        sampler_AS = dset.class_samplers()
        loader = DataLoader(dset, batch_size=bsize, sampler=sampler_AS, num_workers=10)
    else:
        loader = DataLoader(dset, batch_size=bsize, shuffle=True, num_workers=10)
    return loader
    

class AorticStenosisDataset(Dataset):
    def __init__(self, 
                 label_scheme,
                 img_path_dataset,
                 tab_dataset,
                 mat_loader,
                 split: str = 'train',
                 transform: bool = True, normalize: bool = True, 
                 frames: int = 16, resolution: int = 224,
                 flip_rate: float = 0.3, min_crop_ratio: float = 0.8, 
                 hr_mean: float = 4.237, hr_std: float = 0.1885,
                 **kwargs):

        self.hr_mean = hr_mean
        self.hr_srd = hr_std
        self.scheme = label_scheme
        self.mat_loader = mat_loader
        self.dataset = img_path_dataset
        self.tab_dataset = tab_dataset
        self.frames = frames
        self.resolution = (resolution, resolution)
        self.split = split
        self.transform = None
        if transform:
            self.transform = Compose(
                [RandomResizedCrop(size=self.resolution, scale=(min_crop_ratio, 1)),
                 RandomHorizontalFlip(p=flip_rate)]
            )
        self.normalize = normalize

    def class_samplers(self):
        # returns WeightedRandomSamplers
        # based on the frequency of the class occurring
        
        # storing labels as a dictionary will be in a future update
        labels_AS = np.array(self.dataset['as_label'])  
        labels_AS = np.array([self.scheme[t] for t in labels_AS])
        class_sample_count_AS = np.array([len(np.where(labels_AS == t)[0]) 
                                          for t in np.unique(labels_AS)])
        weight_AS = 1. / class_sample_count_AS
        if len(weight_AS) != 4:
            weight_AS = np.insert(weight_AS,0,0)
        samples_weight_AS = np.array([weight_AS[t] for t in labels_AS])
        samples_weight_AS = torch.from_numpy(samples_weight_AS).double()
        sampler_AS = WeightedRandomSampler(samples_weight_AS, len(samples_weight_AS))
        return sampler_AS
        
    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def get_random_interval(vid, length):
        length = int(length)
        start = randint(0, max(0, len(vid) - length))
        return vid[start:start + length]
    
    # expands one channel to 3 color channels, useful for some pretrained nets
    @staticmethod
    def gray_to_gray3(in_tensor):
        # in_tensor is 1xTxHxW
        return in_tensor.expand(3, -1, -1, -1)
    
    # normalizes pixels based on pre-computed mean/std values
    @staticmethod
    def bin_to_norm(in_tensor):
        # in_tensor is 1xTxHxW
        m = 0.099
        std = 0.171
        return (in_tensor-m)/std

    def __getitem__(self, item):
        data_info = self.dataset.iloc[item]

        #get associated tabular data based on echo ID
        study_num = data_info['Echo ID#']
        tab_info = self.tab_dataset.loc[int(study_num)]
        tab_info = torch.tensor(tab_info.values, dtype=torch.float32)

        cine_original = self.mat_loader(data_info['path'])

        window_length = 60000 / (lognormvariate(self.hr_mean, self.hr_srd) * data_info['frame_time'])
        cine = self.get_random_interval(cine_original, window_length)
        cine = resize(cine, (32, *self.resolution))
        cine = torch.tensor(cine).unsqueeze(0)
        
        # storing labels as a dictionary will be in a future update
        # if folder == 'round2':
        #     labels_B = torch.tensor(int(data_info['Bicuspid']))
        labels_AS = torch.tensor(self.scheme[data_info['as_label']])

        if self.transform:
            cine = self.transform(cine)
            
        if self.normalize:
            cine = self.bin_to_norm(cine)

        cine = self.gray_to_gray3(cine)
        cine = cine.float()
        
        ret = (cine, tab_info, labels_AS)

        return ret