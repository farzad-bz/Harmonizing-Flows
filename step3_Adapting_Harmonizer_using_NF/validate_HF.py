# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import random
from validate_sgmentation import validate_segmentation, validate_segmentation_base

## Standard libraries
import os
import math
import time
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

#DataLoader
from hf_dataloader import MedicalImage2DDataset

from NF_model import flow_model
from harmonizer_model import Harmonizer, ConcatELU
from HF_model import FLow_harmonizer



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")
print("Using device", device)


sites = ['CALTECH', 'KKI','PITT', 'NYU']

for source_site in sites:
    for target_site in sites:
        if source_site==target_site:
            continue
        print(f'source:{source_site}  target:{target_site}')

        net_harmonizer = torch.load(f'../checkpoints/UNet2D_harmonizer_{source_site}/model/Best_UNet2D_harmonizer.pkl').cuda()
        flow = flow_model(dequant_mode='variational').cuda()
        
        print("Found pretrained model, loading...")
        ckpt = torch.load(f"../checkpoints/ABIDE-FLOW-{source_site}/ABIDE-Guided-Flow-variational/lightning_logs/version_0/checkpoints/last.ckpt", map_location=device)
        flow.load_state_dict(ckpt['state_dict'])
        flow.eval()
        print('flow model loaded')

        model = FLow_harmonizer(flow, net_harmonizer)
        
        for param in model.flows.parameters():
            param.requires_grad = False
        model.flows.eval()
            
        for param in model.harmonizer.parameters():
            param.requires_grad = False
        model.harmonizer.eval()
            
        root_dir = '../data/'
        df_root_dir = '../data/'
        test_set = MedicalImage2DDataset('test', target_site, root_dir, df_root_path=df_root_dir)
        test_loader = DataLoader(test_set, batch_size=16, num_workers=4, shuffle=True)
        
        with torch.no_grad():
            bpds = []
            for batch_idx, data in enumerate(test_loader):
                img, mask = data
                bpd = model.forward_base(Variable(img).cuda())
                bpds.append(bpd.mean().data.detach().cpu().numpy())
        
        print('before harmonization - BPD:', np.mean(bpds)) 
        output = validate_segmentation_base(source_site=source_site, target_site=target_site)
        print(output)

        with torch.no_grad():
            bpds = []
            for batch_idx, data in enumerate(test_loader):
                img, mask = data
                bpd = model(Variable(img).cuda(), Variable(mask).cuda())
                bpds.append(bpd.mean().data.detach().cpu().numpy())
                
        
        print('only with harmonization network - BPD:', np.mean(bpds)) 
        output = validate_segmentation(model.harmonizer, source_site=source_site, target_site=target_site)
        print(output)

                
        
        for param in model.harmonizer.parameters():
            param.requires_grad = True
        model.harmonizer.train()

        optimizer = optim.Adam(model.harmonizer.parameters(), lr=4e-6)
        model.zero_grad()
        optimizer.zero_grad()
        for i in range(20):
            bpds = []
            for batch_idx, data in enumerate(test_loader):
                img, mask = data
                bpd = model(Variable(img).cuda(), Variable(mask).cuda())
                loss = bpd.mean()
                bpds.append(bpd.data.detach().cpu().numpy())
                loss.backward()
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()
            

            print(f'flow harmonization step {i+1} - BPD: ', np.mean(bpds)) 
            output = validate_segmentation(model.harmonizer, source_site=source_site, target_site=target_site)
            print(output)


            for param in model.harmonizer.parameters():
                param.requires_grad = True
            model.harmonizer.train()


