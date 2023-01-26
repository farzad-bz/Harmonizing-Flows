import numpy as np
import torch
from torch.autograd import Variable

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)



def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()



def getOneHotSegmentation(batch, num_classes):
    backgroundVal = 0

    labels = [i for i in range(num_classes)]



    batch = batch.unsqueeze(dim=1)
    oneHotLabels = torch.cat(tuple([batch == i for i in labels]), dim=1)

    return oneHotLabels.float()



