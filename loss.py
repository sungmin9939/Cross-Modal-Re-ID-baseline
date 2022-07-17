import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

def cal_p2p_loss(X, T, alpha=32, mrg=0.1):
    X = torch.split(X, 4)
    T = torch.split(T, 4)
    P = None
    T_P = None
    for (x,t) in zip(X, T):
        x = torch.mean(x, dim=0, keepdim=True)
        t = torch.unique(t)
        if P is None and T_P is None:
            P = x
            T_P = t
        else:
            P = torch.cat((P, x), dim=0)
            T_P = torch.cat((T_P,t))
    
    P = l2_norm(P)
    cos = F.linear(P, P)
    neg_exp = torch.exp(alpha * (cos + mrg))
    neg_exp = torch.triu(neg_exp, diagonal=1).sum()
    
    neg_term = torch.log(1 + neg_exp)
    
    return neg_term, P, T_P

def cal_p2e_loss(X, P, T, T_P, alpha=32, mrg=0.1):
    P_one_hot = binarize(T, T_P.cpu())
    
    
    cos = F.linear(l2_norm(X), l2_norm(P))
    pos_exp = torch.exp(-alpha * (cos - mrg))
    
    with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
    num_valid_proxies = len(with_pos_proxies)
    
    P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
    pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
    
    return pos_term
    
    



class two_level_Proxy_Anchor(nn.Module):
    def __init__(self, sz_embed, num_classes, mrg=0.1, alpha=32, batch_size=8):
        super().__init__()
        
        self.sz_embed = sz_embed
        self.num_classes = num_classes
        self.mrg = mrg
        self.alpha = alpha
        self.batch_size = batch_size
        
    def forward(self, X, T):
        
        T = torch.split(T, [self.batch_size*8, self.batch_size*4], dim=0)
        X = torch.split(X, [self.batch_size*8, self.batch_size*4], dim=0)
        
        p2p_loss, P, T_P = cal_p2p_loss(X[-1], T[-1])
        p2e_loss = cal_p2e_loss(X[0], P, T[0], T_P)
        
        return p2p_loss, p2e_loss
        
class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss
    
class Proxy_Anchor_linear(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = nn.Linear(sz_embed, nb_classes)
        nn.init.kaiming_normal_(self.proxies.weight, mode='fan_out')
        self.cri = nn.CrossEntropyLoss()

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies.weight

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        loss = self.cri(cos, T)
        
        return loss
    
# class Proxy_Anchor_linear(torch.nn.Module):
#     def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
#         torch.nn.Module.__init__(self)
#         # Proxy Anchor Initialization
#         self.proxies = nn.Linear(sz_embed, nb_classes)
#         nn.init.kaiming_normal_(self.proxies.weight, mode='fan_out')

#         self.nb_classes = nb_classes
#         self.sz_embed = sz_embed
#         self.mrg = mrg
#         self.alpha = alpha
        
#     def forward(self, X, T):
#         P = self.proxies.weight

#         cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
#         P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
#         N_one_hot = 1 - P_one_hot
    
#         pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
#         neg_exp = torch.exp(self.alpha * (cos + self.mrg))

#         with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
#         num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
#         P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
#         N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
#         pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
#         neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
#         loss = pos_term + neg_term     
        
#         return loss

class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct



        
        
# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)


        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct
        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx