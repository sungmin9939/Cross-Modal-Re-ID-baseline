import torch
from utils import grayscale_linearcomb
from model import visible_module, ChannelAttention, SpatialAttnetion, embed_net
from loss import cal_p2e_loss, cal_p2p_loss, binarize, two_level_Proxy_Anchor
import torch.nn as nn
from torch.autograd import Variable


device = 'cuda:0'

net = embed_net(200).to(device)
net = nn.DataParallel(net, device_ids=[0, 1])
input1 = Variable(torch.randn(32,3,288,144).to(device))
input2 = Variable(torch.randn(32,3,288,144).to(device))
input3 = Variable(torch.randn(32,3,288,144).to(device))

feat, out = net(input1, input2, input3)
print(out.shape)

'''
cri = two_level_Proxy_Anchor(512,100).to(device)

input1 = torch.randn(64,512).cuda()
input2 = torch.randn(32,512).cuda()
input3 = torch.Tensor([167, 167, 167, 167, 222, 222, 222, 222, 384, 384, 384, 384, 286, 286,
        286, 286, 289, 289, 289, 289, 147, 147, 147, 147,  31,  31,  31,  31,
        145, 145, 145, 145, 167, 167, 167, 167, 222, 222, 222, 222, 384, 384,
        384, 384, 286, 286, 286, 286, 289, 289, 289, 289, 147, 147, 147, 147,
         31,  31,  31,  31, 145, 145, 145, 145]).cuda()
input4 = torch.Tensor([167, 167, 167, 167, 222, 222, 222, 222, 384, 384, 384, 384, 286, 286,
        286, 286, 289, 289, 289, 289, 147, 147, 147, 147,  31,  31,  31,  31,
        145, 145, 145, 145]).cuda()

p2p, p2e = cri(torch.cat((input1, input2),dim=0), torch.cat((input3, input4)))
'''
'''
loss, P, P_T = cal_p2p_loss(input2, input4)
print(P_T)

loss_p2e = cal_p2e_loss(input1, P, input3, P_T)
print(loss_p2e)
'''
