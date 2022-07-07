import torch
from utils import grayscale_linearcomb
from model import visible_module, ChannelAttention, SpatialAttnetion, embed_net
from loss import cal_p2e_loss, cal_p2p_loss, binarize, two_level_Proxy_Anchor
import torch.nn as nn
from torch.autograd import Variable

# given a tensor A with size (B, 3, H, W), split it into (B/4, 3, H, W)
def split_tensor(A):
    B = A.size(0)
    return A.narrow(0, 0, B//4), A.narrow(0, B//4, B//4), A.narrow(0, B//2, B//4), A.narrow(0, B//4*3, B//4)


input1 = torch.randn(16,3,16,16)
input1 = torch.split(input1, 4, dim=0)
input2 = torch.randn(4,3)
input3 = []
for input in input1:
        temp1 = []
        for i in range(4):
                temp2 = []
                for j in range(3):
                        temp2.append((input[i,j,:,:] * input2[i,j]).unsqueeze(0))
                temp2 = torch.cat(temp2, dim=0)
                temp1.append(temp2.unsqueeze(0))
        temp1 = torch.cat(temp1, dim=0)
        print(temp1.size())
                        
                        
                        