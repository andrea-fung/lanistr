from __future__ import absolute_import

import os
import random
from os.path import join
from typing import Any, Dict, List, Union
from random import seed, randint, lognormvariate
# from dataset.amazon.amazon_utils import get_amazon_transforms
# from dataset.amazon.amazon_utils import get_train_and_test_splits
# from dataset.amazon.amazon_utils import load_multimodal_data
# from dataset.amazon.amazon_utils import preprocess_amazon_tabular_features
import numpy as np
import omegaconf
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip
import transformers
from utils.data_utils import MaskGenerator
from scipy.io import loadmat
from skimage.transform import resize
from dataset.aortic_stenosis.as_utils import load_as_data, preprocess_as_data, fix_leakage


def load_as(
    args: omegaconf.DictConfig
) -> Dict[str, Union[data.Dataset, Dict[str, Any]]]: 
  """Load the Aortic Stenosis dataset.

  Args:
      args: The arguments for the experiment.

  Returns:
      A dictionary containing the train, valid, and test datasets.
  """
  # feature_names = categorical_cols + numerical_cols
  # image_names = ['ImageFileName']
  # text_names = ['Review']

  scheme = {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3}

  image_train, image_val, image_test = get_image_dataset(args, scheme)
  ((tab_train, tab_val, tab_test), (input_dim, cat_idxs, cat_dims)) = get_tab_dataset(args)

  tabular_data_information = { 
      'input_dim': input_dim,
      'cat_idxs': cat_idxs,
      'cat_dims': cat_dims,
    #   'feature_names': feature_names,
    #   'image_names': image_names,
    #   'text_names': text_names,
  }

  dataframes = {
      'image_train': image_train,
      'image_val': image_val,
      'image_test': image_test,
      'tab_train': tab_train, 
      'tab_val': tab_val, 
      'tab_test': tab_test, 
      'tabular_data_information': tabular_data_information,
  }
  dataset = create_multimodal_dataset_from_dataframes(
      args, mat_loader, scheme, dataframes
  )
  return dataset

def create_multimodal_dataset_from_dataframes(
    args: omegaconf.DictConfig,
    mat_loader,
    scheme,
    dataframes: Dict[str, pd.DataFrame]
) -> Dict[str, Union[data.Dataset, Dict[str, Any]]]:
  """Create a multimodal dataset from dataframes.

  Args:
      args: The arguments for the experiment.
      dataframes: The dataframes to use for the dataset.
      tokenizer: The tokenizer to use for the text.

  Returns:
      A dictionary containing the train, valid, and test datasets.
  """

  mm_train = ASImageTabular(
      args=args,
      image_df=dataframes['image_train'],
      tab_df=dataframes['tab_train'],
      get_video=False,
      image=args.image,
      tab=args.tab,
      label_scheme=scheme,
      mat_loader=mat_loader,
      transform=True,
      normalize=True
  )

  mm_test = ASImageTabular(
      args=args,
      image_df=dataframes['image_test'],
      tab_df=dataframes['tab_test'],
      get_video=True,
      image=args.image,
      tab=args.tab,
      label_scheme=scheme,
      mat_loader=mat_loader,
      transform=False,
      normalize=True
  )

  mm_val = ASImageTabular(
      args=args,
      image_df=dataframes['image_val'],
      tab_df=dataframes['tab_val'],
      get_video=True,
      image=args.image,
      tab=args.tab,
      label_scheme=scheme,
      mat_loader=mat_loader,
      transform=False,
      normalize=True
  )

  return {
      'train': mm_train,
      'valid': mm_val,
      'test': mm_test,
      'tabular_data_information': dataframes['tabular_data_information'],
  }

class ASImageTabular(data.Dataset): 
  """AS dataset with image and tabular data."""

  def __init__(
      self,
      args: omegaconf.DictConfig,
      image_df: pd.DataFrame,
      tab_df: pd.DataFrame,
      # feature_names: List[str],
      # image_names: List[str],
      get_video: bool,
      image: bool,
      tab: bool,
      label_scheme,
      mat_loader,
      transform: bool,
      normalize: bool
  ):
    """Initialize the ASImageTabular dataset.

    Args:
        args: The arguments for the experiment.
        df: The dataframe to use for the dataset.
        transform: The transform to use for the images.
        feature_names: The names of the features columns.
        image_names: The names of the image columns.
        image: Whether to use images.
        tab: Whether to use tabular data.
    """
    self.args = args
    self.image_df = image_df
    self.tab_df = tab_df
    self.transform = transform
    # if tab:
    #   self.features = self.df[feature_names].values
    self.mask_generator = MaskGenerator(
        input_size=args.image_size,
        mask_patch_size=args.mask_patch_size,
        model_patch_size=args.model_patch_size,
        mask_ratio=args.image_masking_ratio,
    )
    self.mat_loader = mat_loader
    self.get_video = get_video
    self.image = image
    self.tab = tab
    self.scheme = label_scheme
    self.transform = None
    if transform:
        self.transform = Compose(
            [RandomResizedCrop(size=(args.image_size, args.image_size), scale=(args.min_crop_ratio, 1)),
                RandomHorizontalFlip(p=args.flip_rate)]
        )
    self.normalize = normalize

  def class_samplers(self):
    # returns WeightedRandomSamplers
    # based on the frequency of the class occurring
    
    # storing labels as a dictionary will be in a future update
    labels_AS = np.array(self.image_df['as_label'])  
    labels_AS = np.array([self.scheme[t] for t in labels_AS])
    class_sample_count_AS = np.array([len(np.where(labels_AS == t)[0]) 
                                    for t in np.unique(labels_AS)])
    weight_AS = 1. / class_sample_count_AS
    if len(weight_AS) != 4:
        weight_AS = np.insert(weight_AS,0,0)
    samples_weight_AS = np.array([weight_AS[t] for t in labels_AS])
    samples_weight_AS = torch.from_numpy(samples_weight_AS).double()
    sampler_AS = data.WeightedRandomSampler(samples_weight_AS, len(samples_weight_AS))
    return sampler_AS

  @staticmethod
  def get_random_interval(vid, length):
    length = int(length)
    start = randint(0, max(0, len(vid) - length))
    return vid[start:start + length]

# expands one channel to 3 color channels, useful for some pretrained nets
  @staticmethod
  def gray_to_gray3(in_tensor):
    # in_tensor is 1xTxHxW
    return in_tensor.expand(-1, 3, -1, -1)

# normalizes pixels based on pre-computed mean/std values
  @staticmethod
  def bin_to_norm(in_tensor):
    # in_tensor is 1xTxHxW
    m = 0.099
    std = 0.171
    return (in_tensor-m)/std

  def __getitem__(self, index: int):
    """Get the item at the given index.

    Args:
        index: The index of the item to get.

    Returns:
        The item at the given index.
    """
    data_info = self.image_df.iloc[index]

    item = {}

    #get associated tabular data based on echo ID
    study_num = data_info['Echo ID#']

    # image
    if self.image:
      pixel_values = []
      bool_masked_positions = []

      cine_path = data_info['path']

      if isinstance(cine_path, str):
        cine_original = self.mat_loader(cine_path)
        window_length = 60000 / (lognormvariate(self.args.hr_mean, self.args.hr_std) * data_info['frame_time'])
        cine = self.get_random_interval(cine_original, window_length)
        if self.get_video:
            cine = resize(cine, (32, self.args.image_size, self.args.image_size)) #[1,32,224,224]
        else:
            frame_choice = np.random.randint(0, cine.shape[0], 1)
            cine = cine[frame_choice, :, :]
            cine = resize(cine, (1, self.args.image_size, self.args.image_size)) #[1,1,224,224]

        cine = torch.tensor(cine).unsqueeze(0)
        if self.transform:
            cine = self.transform(cine)
            
        if self.normalize:
            cine = self.bin_to_norm(cine)

        cine = self.gray_to_gray3(cine) #[32,3,224,224] or [1, 3, 224, 224]
        cine = cine.float()
        
        pixel_values.append(cine)
      else:
        if self.get_video:
            pixel_values.append(torch.zeros(
            size=(32, 3, self.args.image_size, self.args.image_size),
            dtype=torch.float,
            ))
        else:
            pixel_values.append(torch.zeros(
            size=(3, self.args.image_size, self.args.image_size),
            dtype=torch.float,
            ))
    
      bool_masked_positions.append(self.mask_generator())
      # pixel_values has shape (image_num, channel, width, height)
      item['pixel_values'] = torch.stack(pixel_values).squeeze(0)
      # bool_masked_positions has shape (image_num, model_patch_size**2)
      item['bool_masked_positions'] = torch.stack(bool_masked_positions)

    # tabular
    if self.tab:
      tab_info = self.tab_df.loc[int(study_num)]
      tab_info = torch.tensor(tab_info.values, dtype=torch.float32) #[B, D]
      item['features'] = tab_info
      # item['features'] = torch.tensor(
      #     np.vstack(self.features[index]).astype(np.float32)
      # ).squeeze(1)

    # ground truth label if finetuning
    if self.args.task == 'finetune':
      item['labels'] = torch.tensor(self.scheme[data_info['as_label']], dtype=torch.long)

    return item

  def __len__(self) -> int:
    """Get the length of the dataset.

    Returns:
        The length of the dataset.
    """
    return len(self.image_df)

def get_image_dataset(args: omegaconf.DictConfig, scheme): 
    raw_dataset = pd.read_csv(args.img_path_dataset)
    dataset = raw_dataset.copy()
        
    # append dataset root to each path in the dataframe
    dataset['path'] = dataset['path'].map(lambda x: join(args.dataset_root, x))
       
    # remove unnecessary columns in 'as_label' based on label scheme
    dataset = dataset[dataset['as_label'].isin(scheme.keys() )]

    # Take train/test/val
    train_dataset = dataset[dataset['split'] == 'train']
    val_dataset = dataset[dataset['split'] == 'val']
    test_dataset = dataset[dataset['split'] == 'test']
    
    #Fix data leakage 
    train_set = fix_leakage(df=raw_dataset, df_subset=train_dataset, split='train')
    val_set = fix_leakage(df=raw_dataset, df_subset=val_dataset, split='val')
    test_set = fix_leakage(df=raw_dataset, df_subset=test_dataset, split='test')
  
    return train_set, val_set, test_set

def get_tab_dataset(args: omegaconf.DictConfig):  
  drop_cols = []
  ((tab_train, tab_val, tab_test), (input_dim, cat_idxs, cat_dims)) = load_as_data(csv_path = args.tab_path_dataset,
                                                                                    drop_cols = drop_cols,
                                                                                    num_ex = None,
                                                                                    scale_feats = args.scale_feats)
  #perform imputation 
  categorical_cols = tab_train.columns[np.array(cat_idxs)+1].to_list()
  train_set, val_set, test_set, all_cols = preprocess_as_data(tab_train, tab_val, tab_test, categorical_cols)
  
  return ((train_set, val_set, test_set), (input_dim, cat_idxs, cat_dims))

def mat_loader(path):
    mat = loadmat(path)
    if 'cine' in mat.keys():    
        return loadmat(path)['cine']
    if 'cropped' in mat.keys():    
        return loadmat(path)['cropped']