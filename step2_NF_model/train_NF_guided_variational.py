###The basline of the codes are adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html


## Standard libraries
import os
import math
import time
import numpy as np

## Progress bar
from tqdm.notebook import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import Callback

#DataLoader
from flow_guided_dataloader import MedicalImage2DDataset

#NF model
from NF_model import flow_model

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


device = torch.device("cuda:0")
print("Using device", device)

for site in ['CALTECH', 'KKI', 'PITT', 'NYU']:

    root_dir = '../data/'
    df_root_dir = '../data/'
    CHECKPOINT_PATH = f'../checkpoints/ABIDE-FLOW-{site}'



    train_set = MedicalImage2DDataset('train', site, root_dir, df_root_path=df_root_dir, full=True)
    train_loader = data.DataLoader(train_set, batch_size=24, shuffle=True, pin_memory=True, num_workers=8)

    val_set = MedicalImage2DDataset('val', site, root_dir, df_root_path=df_root_dir, full=True)
    val_loader = data.DataLoader(val_set, batch_size=24, shuffle=False, pin_memory=True, num_workers=8)

    class PrintCallback(Callback):
        def on_epoch_end(self, trainer, pl_module):
            print("")

    def train_flow(flow, model_name="ABIDE-Guided-Flow-variational"):
        # Create a PyTorch Lightning trainer
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
                            gpus=1 if torch.cuda.is_available() else 0,
                            #accelerator='dp',
                            max_epochs=1600,
                            gradient_clip_val=1.0,
                            callbacks=[PrintCallback(),
                                        ModelCheckpoint(save_weights_only=True, save_top_k=-1, every_n_epochs=250, save_last=True),
                                        LearningRateMonitor("epoch")],
                            check_val_every_n_epoch=200)
        trainer.logger._log_graph = True
        trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

        result = None

        # Check whether pretrained model exists. If yes, load it and skip training)
        
        print("Start training", model_name)
        trainer.fit(flow, train_loader, val_loader)

        # Test best model on validation and test set if no result has been found
        # Testing can be expensive due to the importance sampling.
        start_time = time.time()
        val_result = trainer.test(flow, val_loader, verbose=False)
        duration = time.time() - start_time
        result = {"val": val_result, "time": duration / len(val_loader) / flow.import_samples}

        print(result)

    train_flow(flow_model(dequant_mode='variational'))