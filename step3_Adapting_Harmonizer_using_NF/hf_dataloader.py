from operator import index
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
import warnings
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

warnings.filterwarnings("ignore")

def make_2Ddataset(root_dir, df_root_path, mode, site):
    """
    Args:
        root_dir : path to volumes 
        df_root_path (string): dataframe directory containing csv files
        mode (string): 'train', 'val', 'test'
        num_classes : 10 or 88 for DHCP data
    """
    assert mode in ['train', 'val', 'test']
    items = []
    

    df = pd.read_csv(f'{df_root_path}/ABIDE-slices-{mode}-dataframe.csv', index_col=0)
    df = df.query('SITE==@site').reset_index(drop=True)

    data_paths = df['T1'].values
    mask_paths = df['brainmask'].values
    
    names = df['FILE_ID'].values

    slices = df['slice_num'].values

    for it_im, it_mk, it_slice, it_nm in zip(data_paths, mask_paths, slices, names):
        item = (os.path.join(root_dir, it_im), os.path.join(root_dir, it_mk), it_slice, it_nm)
        items.append(item)

    return items 


class MedicalImage2DDataset(Dataset):
    """ABIDE dataset."""

    def __init__(self, mode, site, root_dir='.', normalization=None, df_root_path = '.', full=False):
        """
        Args:
            mode: 'train','val','test'
            root_dir (string): Directory with all the volumes.
            transform (callable, optional): Optional transform to be applied on a sample.
            df_root_path (string): dataframe directory containing csv files
        """
        self.normalization = normalization
        self.root_dir = root_dir
        self.mode = mode
        self.site = site
        self.items = make_2Ddataset(root_dir, df_root_path, mode, site, full=full)


    def transform_volume(self, x):
        if len(x.shape) == 2:
            x = np.expand_dims(x, -1)
        x = torch.from_numpy(x.transpose((-1, 0 , 1)))
        return x

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        data_path, mask_path, slice_num, name = self.items[index]
        img = np.load(data_path)
        mask = np.load(mask_path)
        mask = (mask>0).astype(np.int32)
        if self.mode == 'train':
            start_x = np.random.randint(16)
            start_y = np.random.randint(16)
            img = img[8+start_x:start_x+232, 8+start_y:start_y+232]
            mask = mask[8+start_x:start_x+232, 8+start_y:start_y+232]
        
        else:
            start_x = np.random.randint(16)
            start_y = np.random.randint(16)
            img = img[8+start_x:start_x+232, 8+start_y:start_y+232]
            mask = mask[8+start_x:start_x+232, 8+start_y:start_y+232]
            
        img[mask==0] = 0
        img = self.transform_volume(img.astype(np.float32))
        mask = self.transform_volume(mask.astype(np.float32))
        return img, mask