
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
import os
import torch.nn.functional as F
from harmonizer_dataloader import MedicalImage2DDataset
from harmonizer_model import Harmonizer
from progressBar import printProgressBar

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print('-' * 40)
print('~~~~~~~~  Starting the training... ~~~~~~')
print('-' * 40)

lr = 0.001
epoch = 200 
seed = 42

root_dir = '../data/'
df_root_dir = '../data/'
modelName = 'UNet2D_harmonizer'

for site in ['PITT', 'NYU', 'CALTECH', 'KKI']:

    main_dir = f'../checkpoints/UNet2D_harmonizer_{site}/'
    model_dir = main_dir + 'model/'

    # set random seed for all gpus
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    train_set = MedicalImage2DDataset('train', site, root_dir, df_root_path=df_root_dir)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

    val_set = MedicalImage2DDataset('val', site, root_dir, df_root_path=df_root_dir)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)



    def to_var(x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)



    print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")


    netG = Harmonizer(1)
    MSE_loss = nn.MSELoss(reduction='mean')

    if torch.cuda.is_available():
        netG.cuda()
        netG = nn.DataParallel(netG)
        MSE_loss.cuda()

        
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.9, 0.99), amsgrad=False)



    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    Losses_total = []
    Losses_val_total = []
    Bestloss = 1e6
    BestEpoch = 0

    for i in range(epoch):
        loss_total = []
        
        for j, data in enumerate(train_loader):
            
            image = data[0]
            orig_image = data[1]

            netG.train()
            optimizerG.zero_grad()
            MRI = to_var(image)
            orig_image = to_var(orig_image)
            
            ################### Train ###################
            netG.zero_grad()

            deepSupervision = False
            multiTask = False

            reconstructed_image = netG(MRI)
            
            loss = MSE_loss(reconstructed_image, orig_image)
            loss.backward()
            optimizerG.step()
                
            loss_total.append(loss.cpu().data.numpy())

            printProgressBar(j, len(train_loader),
                                prefix="[Training] Epoch: {} ".format(i),
                                length=15,
                                suffix=" loss_total: {:.4f}".format(loss.data))

        Losses_total.append(np.mean(loss_total))

        printProgressBar(j, len(train_loader),
                            prefix="[Training] Epoch: {} ".format(i),
                            length=15,
                            suffix=" loss_total: {:.4f}".format(np.mean(loss_total)))

        directory = main_dir + 'Statistics/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(os.path.join(directory, 'train-Losses.npy'), Losses_total)


        loss_val_total = []
        for j, data in enumerate(val_loader):
            image = data[0]
            orig_image = data[1]

            netG.eval()
            optimizerG.zero_grad()
            MRI = to_var(image)
            orig_image = to_var(orig_image)
            
            ################### Train ###################
            netG.zero_grad()
            reconstructed_image = netG(MRI)
            loss_val = MSE_loss(reconstructed_image, orig_image)

            # Save for plots
            loss_val_total.append(loss_val.cpu().data.numpy())

            printProgressBar(j , len(val_loader),
                                prefix="[validation] Epoch: {} ".format(i),
                                length=15,
                                suffix=" loss_val_total: {:.4f}".format(loss_val.data))



        Losses_val_total.append(np.mean(loss_val_total))

        np.save(os.path.join(directory, 'val-Losses.npy'), Losses_val_total)

        printProgressBar(j, len(val_loader),
                            prefix="[validation] Epoch: {} ".format(i),
                            length=15,
                            suffix=" loss_val_total: {:.4f}".format(np.mean(loss_val_total)))


        Currentloss = np.mean(loss_val_total)

        if Currentloss < Bestloss:
            Bestloss = Currentloss
            BestEpoch = i

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Saving best model..... ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(netG, os.path.join(model_dir, "Best_" + modelName + ".pkl"))

        print("### Best Loss(mean): at epoch {} with (loss): {:.4f}  ###".format(BestEpoch, Bestloss))
        print(' ')
        if i % 30 == 29 :
            for param_group in optimizerG.param_groups:
                    param_group['lr'] = lr/2


