import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from utils import *
import medpy

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class Test_Dataset(Dataset):
    """ABIDE dataset."""

    def __init__(self, site, image_id, full=True):
        
        self.root_dir = '../data'
        df_root_dir = '../data/'
        df = pd.read_csv(f'{df_root_dir}/ABIDE-slices-test-dataframe.csv', index_col=0)
        df = df.query('SITE==@site').reset_index(drop=True)
        self.sub_df = df.query('FILE_ID==@image_id').reset_index(drop=True)
        
    def transform_volume(self, x):
        if len(x.shape) == 2:
            x = np.expand_dims(x, -1)
        x = torch.from_numpy(x.transpose((-1, 0 , 1)))
        return x

    def __len__(self):
        return len(self.sub_df)

    def __getitem__(self, index):
        row = self.sub_df.iloc[index]
        img = np.load(self.root_dir + row['T1']).astype(np.uint8)[16:240, 16:240]
        mask = np.load(self.root_dir + row['brainmask']).astype(np.uint8)[16:240, 16:240]
        mask = (mask>0).astype(np.int32)
        img[mask==0] = 0
        seg = np.load(self.root_dir + row['segmentation'])[16:240, 16:240]
        img = self.transform_volume(img.astype(np.float32))
        return img, mask, seg


def validate_segmentation(harmonizer_net, source_site='CALTECH', target_site='NYU', num_classes=15):
    df_root_dir = '../data/'
    segmentation_net = torch.load(f'../checkpoints/segmentation_network_{source_site}.pkl')
    segmentation_net.eval()
    harmonizer_net.eval()
    softMax = nn.Softmax(dim=1)
    with torch.no_grad():
        df = pd.read_csv(f'{df_root_dir}/ABIDE-slices-test-dataframe.csv', index_col=0).query('SITE==@target_site').reset_index(drop=True)
        image_ids = np.unique(df['FILE_ID'].values)
        Dice_total = []
        Hd95_total = []
        flag = False
        output_entropy = 0
        for image_id in image_ids:
            test_set = Test_Dataset(target_site, image_id)
            test_loader = DataLoader(test_set, batch_size=32, num_workers=4, shuffle=False)
            prediction_3d = []
            segmentation_3d = []
            for data,mask,seg  in test_loader:
                reconstructed_data = harmonizer_net(Variable(data).cuda())
                reconstructed_data = reconstructed_data.detach().cpu().numpy().astype(np.int32)
                reconstructed_data = np.clip(reconstructed_data, 0,255).astype(np.int32)
                reconstructed_data[mask.unsqueeze(1)==0] = 0
                reconstructed_data = reconstructed_data.astype(np.float32)/255.0
                reconstructed_data = Variable(torch.from_numpy(reconstructed_data)).cuda()
                segmentation_prediction = segmentation_net(reconstructed_data)
                output_entropy += np.sum(softmax_entropy(segmentation_prediction).detach().cpu().numpy())
                predClass_y = softMax(segmentation_prediction)
                segmentation_seg_ones = getOneHotSegmentation(seg, num_classes).detach().numpy()
                segmentation_prediction_ones = predToSegmentation(predClass_y).detach().cpu().numpy()
                prediction_3d.append(segmentation_prediction_ones)
                segmentation_3d.append(segmentation_seg_ones)
            prediction_3d = np.concatenate(prediction_3d, axis=0)
            segmentation_3d = np.concatenate(segmentation_3d, axis=0)
            
            binary_dcs = []
            binary_hd95s = []
            for j in range(1, num_classes):
                dc = medpy.metric.binary.dc(prediction_3d[:,j], segmentation_3d[:,j])
                binary_dcs.append(dc)
                if np.sum(prediction_3d[:,j]) == 0:
                    wrong_seg = np.zeros_like(segmentation_3d[:,j])
                    wrong_seg[0,0] = 1
                    hd = medpy.metric.binary.hd95(wrong_seg, segmentation_3d[:,j])
                    binary_hd95s.append(hd)                        
                else:
                    hd = medpy.metric.binary.hd95(prediction_3d[:,j], segmentation_3d[:,j])
                    binary_hd95s.append(hd)
            
            Dice_total.append(np.mean(binary_dcs))
            Hd95_total.append(np.mean(binary_hd95s))

    return '{:.1f}±{:.1f} & {:.1f}±{:.1f}-------Emtropy:{}'.format(100*np.mean(Dice_total),  100*np.std(Dice_total),  np.mean(Hd95_total),  np.std(Hd95_total), output_entropy)

            
def validate_segmentation_base(source_site='CALTECH', target_site='NYU', num_classes=15):
    df_root_dir = '../data/'
    segmentation_net = torch.load(f'../checkpoints/segmentation_network_{source_site}.pkl')
    segmentation_net.eval()
    softMax = nn.Softmax(dim=1)
    binary_dcs = []
    with torch.no_grad():
        df = pd.read_csv(f'{df_root_dir}/ABIDE-slices-test-dataframe.csv', index_col=0).query('SITE==@target_site').reset_index(drop=True)
        image_ids = np.unique(df['FILE_ID'].values)
        Dice_total = []
        Hd95_total = []
        flag = False
        output_entropy = 0
        for image_id in image_ids:
            prediction_3d = []
            segmentation_3d = []
            test_set = Test_Dataset(target_site, image_id)
            test_loader = DataLoader(test_set, batch_size=32, num_workers=4, shuffle=False)
            for data,mask,seg  in test_loader:
                segmentation_prediction = segmentation_net(Variable(data/255.0).cuda())
                output_entropy += np.sum(softmax_entropy(segmentation_prediction).detach().cpu().numpy())
                predClass_y = softMax(segmentation_prediction)
                segmentation_seg_ones = getOneHotSegmentation(seg, num_classes).detach().numpy()
                segmentation_prediction_ones = predToSegmentation(predClass_y).detach().cpu().numpy()
                prediction_3d.append(segmentation_prediction_ones)
                segmentation_3d.append(segmentation_seg_ones)
            prediction_3d = np.concatenate(prediction_3d, axis=0)
            segmentation_3d = np.concatenate(segmentation_3d, axis=0)
            binary_dcs = []
            binary_hd95s = []
            for j in range(1, num_classes):
                dc = medpy.metric.binary.dc(prediction_3d[:,j], segmentation_3d[:,j])
                binary_dcs.append(dc)
                if np.sum(prediction_3d[:,j]) == 0:
                    wrong_seg = np.zeros_like(segmentation_3d[:,j])
                    wrong_seg[0,0] = 1
                    hd = medpy.metric.binary.hd95(wrong_seg, segmentation_3d[:,j])
                    binary_hd95s.append(hd)                        
                else:
                    hd = medpy.metric.binary.hd95(prediction_3d[:,j], segmentation_3d[:,j])
                    binary_hd95s.append(hd)
            
            Dice_total.append(np.mean(binary_dcs))
            Hd95_total.append(np.mean(binary_hd95s))

    return '{:.1f}±{:.1f} & {:.1f}±{:.1f}-------Emtropy:{}'.format(100*np.mean(Dice_total),  100*np.std(Dice_total),  np.mean(Hd95_total),  np.std(Hd95_total), output_entropy)
