
import torch
import torch.nn as nn
from torch.nn import init

from torchvision.models import resnet50


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)



class visible_module(nn.Module):
    def __init__(self, arch='resnet50', pretrained=True):
        super(visible_module, self).__init__()

        self.model = resnet50(pretrained)
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50', pretrained=True):
        super(thermal_module, self).__init__()

        self.model = resnet50(pretrained)
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        return x
    
class synthetic(nn.Module): ## modified
    def __init__(self, arch='resnet50', pretrained=True):
        super(synthetic, self).__init__()

        self.model = resnet50(pretrained)
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        return x

class base_resnet(nn.Module):
    def __init__(self, embedding_size, arch='resnet50', pretrained=True, is_norm=True, bn_freeze=True):
        super(base_resnet, self).__init__()
        
        self.model = resnet50(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)
        
        # self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        # self._initialize_weights()
        
    
    def forward(self, x):
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        '''
        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)


        x = max_x + avg_x
        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)
        '''
        return x
    
    # def _initialize_weights(self):
    #     init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
    #     init.constant_(self.model.embedding.bias, 0)
        
class base_resnet_v2(nn.Module):
    def __init__(self, embedding_size, arch='resnet50', pretrained=True, is_norm=True, bn_freeze=True):
        super(base_resnet_v2, self).__init__()
        
        self.model = resnet50(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)
        
        
        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
                    
    
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output
    
    def forward(self, x):
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        

        x = avg_x
        x = x.view(x.size(0), -1)
        
        if self.is_norm:
            x = self.l2_norm(x)
        
        return x
    

class ChannelAttention(nn.Module): ##modified
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        self.apply(weights_init_kaiming)
        
    def forward(self, x):
        avg_result = self.avgpool(x)
        avg_out = self.se(avg_result)
        output = self.sigmoid(avg_out)
        
        return output
    
class SpatialAttnetion(nn.Module): ##modified
    def __init__(self, channel, channel_reduction=2, kernel_size=3, dilations=[1,2,3]):
        super().__init__()
        self.channel_reduction = channel//channel_reduction
        self.conv1 = nn.Conv2d(channel, self.channel_reduction, 1)
        self.conv2 = nn.Conv2d(self.channel_reduction, self.channel_reduction, 3, dilation=dilations[0], padding=1)
        self.conv3 = nn.Conv2d(self.channel_reduction, self.channel_reduction, 3, dilation=dilations[1], padding=2)
        self.conv4 = nn.Conv2d(self.channel_reduction, self.channel_reduction, 3, dilation=dilations[2], padding=3)
        
        self.conv5 = nn.Conv2d(self.channel_reduction*4, 1, 1)
        
        self.apply(weights_init_kaiming)
        
    def forward(self, x):
        conv1_result = self.conv1(x)
        conv2_result = self.conv2(conv1_result)
        conv3_result = self.conv3(conv1_result)
        conv4_result = self.conv4(conv1_result)
        
        
        conv_result = torch.cat((conv1_result, conv2_result, conv3_result, conv4_result), dim=1)
        
        output = self.conv5(conv_result)
        
        return output

class Synthesizing(nn.Module):
    def __init__(self, num_samples=4):
        self.weights = nn.Parameter(torch.randn(4,3))
        self.num_samples = num_samples
    def forward(self, X):
        result = []
        X = torch.split(X, self.num_samples, dim = 0)
        for x in X:
            temp1 = []
            for i in range(self.num_samples):
                temp2 = []
                for j in range(3):
                    temp2.append((x[i,j,:,:] * self.weights[i,j]).unsqueeze(0))
                temp2 = torch.cat(temp2, dim=0)
                temp1.append(temp2.unsqueeze(0))
            temp1 = torch.cat(temp1, dim=0)
            temp1 = torch.sum(temp1, dim=0, keepdim=True)
            result.append(temp1)
        result = torch.cat(result, dim=0)
        
        return result
            
        
         
        
        
        
        

class embed_net(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50', embed_size = 512, syn=False, local_attn=False, proxy=False):
        super(embed_net, self).__init__()
        self.syn = syn
        self.local_attn = local_attn
        self.proxy = proxy
        self.embedding_size = embed_size

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        if self.syn:
            self.relu = nn.ReLU()
            self.incep = nn.Conv2d(64,64,1)
            # self.synthetic_module = synthetic(arch=arch)
            # self.synthesizing_module = Synthesizing()
            
            
        self.base_resnet = base_resnet(embedding_size=self.embedding_size, arch=arch)
        self.non_local = no_local
        if self.non_local =='on':
            layers=[3, 4, 6, 3]
            non_layers=[0,2,3,0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])


        pool_dim = 2048
        
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        
        self.ca = ChannelAttention(64) 
        self.sa = SpatialAttnetion(64) 

        self.classifier = nn.Linear(pool_dim, self.embedding_size, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        
        self.gm_pool = gm_pool

    def forward(self, x1=None, x2=None, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            '''
            if self.syn:
                x3 = self.relu(x1 * x2)
                x3 = self.incep(x3)
            '''    
            
                        
            '''
            if self.local_attn:            
                x1_ca = x1 * self.ca(x1) + x1
                x2_ca = x2 * self.ca(x2) + x2
                
                x1 = x1_ca * self.sa(x1) + x1_ca
                x2 = x2_ca * self.sa(x2) + x2_ca
            '''
            
            if self.syn:
                x = torch.cat((x1, x2, x3), 0)
            else:
                x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
            if self.syn:    
                pass
                # x_ca = x * self.ca(x) + x
                # x = x_ca * self.sa(x) + x_ca
        elif modal == 2:
            x = self.thermal_module(x2)
            if self.syn:
                pass
                # x_ca = x * self.ca(x) + x
                # x = x_ca * self.sa(x) + x_ca

        # shared block
        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            for i in range(len(self.base_resnet.base.layer1)):
                x = self.base_resnet.base.layer1[i](x)
                if i == self.NL_1_idx[NL1_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1
            # Layer 2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            # Layer 3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            # Layer 4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
        else:
            x = self.base_resnet(x)
        if self.gm_pool  == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        

        feat = self.bottleneck(x_pool)

        if self.training:
            return x_pool, self.classifier(feat) 
        else:
            return self.l2norm(x_pool), self.l2norm(self.classifier(feat))


class embed_net_v2(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50', embed_size = 512, syn=False, local_attn=False, proxy=False):
        super(embed_net_v2, self).__init__()
        self.syn = syn
        self.local_attn = local_attn
        self.proxy = proxy
        self.embedding_size = embed_size
        self.pool_dim = 2048
        self.class_num = class_num

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        
            
        self.base_resnet = base_resnet_v2(embedding_size=self.embedding_size, arch=arch)
        self.non_local = no_local
        
        # self.rgb_linear = nn.Linear(self.pool_dim, self.class_num)
        # self.ir_linear = nn.Linear(self.pool_dim, self.class_num)
        self.mono_linear = nn.Linear(self.pool_dim, self.class_num)

        
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(self.pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        
        
        
        self.ca = ChannelAttention(64) 
        self.sa = SpatialAttnetion(64) 

        #self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self._initialize_weights()
        
        self.gm_pool = gm_pool

    def _initialize_weights(self):
        # init.kaiming_normal_(self.rgb_linear.weight, mode='fan_out')
        # init.constant_(self.rgb_linear.bias, 0)
        
        # init.kaiming_normal_(self.ir_linear.weight, mode='fan_out')
        # init.constant_(self.ir_linear.bias, 0)
        
        init.kaiming_normal_(self.mono_linear.weight, mode='fan_out')
        init.constant_(self.mono_linear.bias, 0)
        
        
    def forward(self, x1=None, x2=None, modal=0):
        
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
            
        elif modal == 2:
            x = self.thermal_module(x2)
        
        x = self.base_resnet(x)
                
        x_pool = x
        
        feat = self.bottleneck(x_pool)

        if self.training:
            return self.mono_linear(feat)
            
        else:
            return x_pool, feat