from __future__ import print_function
import argparse
from doctest import FAIL_FAST
from pickletools import optimize

import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
from loss import OriTripletLoss, TripletLoss_WRT, two_level_Proxy_Anchor, Proxy_Anchor, Proxy_Anchor_linear
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')


parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')

parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')

parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=16, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')

parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

parser.add_argument('--warm', default=0, type=int)

parser.add_argument('--lr', default=1e-4 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--weight-decay', default = 1e-4, type =float, help = 'Weight decay setting')
parser.add_argument('--lr-decay-step', default = 10, type =int, help = 'Learning decay step setting')
parser.add_argument('--lr-decay-gamma', default = 0.5, type =float, help = 'Learning decay gamma setting')
parser.add_argument('--bn-freeze', default = 1, type = int, help = 'Batch normalization parameter freeze')
parser.add_argument('--l2-norm', default = 1, type = int, help = 'L2 normlization')


parser.add_argument('--optim', default='Adam', type=str, help='optimizer')
parser.add_argument('--method', default='base', type=str,
                    metavar='m', help='method type: base or agw')
parser.add_argument('--local-attn', default=False, type=bool)
parser.add_argument('--proxy', default=False, type=bool)
parser.add_argument('--exp', default=4, type=int)
args = parser.parse_args()
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = './dataset/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = '../Datasets/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset


suffix = dataset
if args.method=='agw':
    suffix = suffix + '_agw_p{}_n{}_lr_{}_seed_{}_local_attn_{}_proxy_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed, args.local_attn, args.proxy)
else:
    suffix = suffix + '_base_p{}_n{}_lr_{}_seed_{}_local_attn_{}_proxy_{}_exp_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed, args.local_attn, args.proxy, args.exp)

print(suffix)

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
if args.method =='base':
    net = embed_net(n_class, no_local= 'off', gm_pool =  'off', arch=args.arch, local_attn=args.local_attn, proxy=args.proxy)
else:
    net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch, local_attn=args.local_attn, proxy=args.proxy)
net.to(device)
net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])

cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss().to(device)
if args.method == 'agw':
    if args.proxy:
        criterion_proxy_rgb = Proxy_Anchor(395, 512).to(device)
        criterion_proxy_ir = Proxy_Anchor(395, 512).to(device)
    else:
        criterion_tri = TripletLoss_WRT().to(device)
else:
    if args.proxy:
        criterion_proxy_rgb = Proxy_Anchor(395, 512).to(device)
        criterion_proxy_ir = Proxy_Anchor(395, 512).to(device)
    else:        
        loader_batch = args.batch_size * args.num_pos
        criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)




if args.optim == 'sgd':
    ignored_params = list(map(id, net.module.bottleneck.parameters())) \
                    + list(map(id, net.module.ca.parameters()))\
                    + list(map(id, net.module.sa.parameters()))\
                    + list(map(id, net.module.classifier.parameters()))\
                    + list(map(id, net.module.embedding.parameters()))\
                        
    base_params = filter(lambda p: id(p) not in ignored_params, net.module.parameters())
    
    param_groups = [
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.module.bottleneck.parameters(), 'lr': args.lr},
    ]
    
    if args.local_attn:
        param_groups.append({'params': net.module.ca.parameters(), 'lr': args.lr})
        param_groups.append({'params': net.module.sa.parameters(), 'lr': args.lr})
    if args.proxy:
        param_groups.append({'params': criterion_proxy_ir.parameters(), 'lr': args.lr})
        param_groups.append({'params': criterion_proxy_rgb.parameters(), 'lr': args.lr})
        param_groups.append({'params': net.module.embedding.parameters(), 'lr': args.lr})
    else:
        param_groups.append({'params': net.module.classifier.parameters(), 'lr': args.lr})
        
    
    optimizer = optim.SGD(param_groups, weight_decay=5e-4, momentum=0.9, nesterov=True)

if args.optim == 'Adam':
    param_groups = [
        {'params': list(set(net.module.parameters()).difference(set(net.module.base_resnet.model.embedding.parameters())))},
        {'params': net.module.base_resnet.model.embedding.parameters(), 'lr':float(args.lr) * 1},
    ]
    
    if args.local_attn:
        param_groups.append({'params': net.module.ca.parameters(), 'lr': args.lr})
        param_groups.append({'params': net.module.sa.parameters(), 'lr': args.lr})
    if args.proxy:
        param_groups.append({'params': criterion_proxy_ir.parameters(), 'lr': float(args.lr) * 100})
        param_groups.append({'params': criterion_proxy_rgb.parameters(), 'lr': float(args.lr) * 100})
        # param_groups.append({'params': net.module.embedding.parameters(), 'lr': args.lr * 10})
    else:
        param_groups.append({'params': net.module.classifier.parameters(), 'lr': args.lr})
    
    
    optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay, lr=float(args.lr))

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma = args.lr_decay_gamma)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def train(epoch):

    current_lr = optimizer.param_groups[0]['lr']
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    p2p_loss = AverageMeter()
    p2e_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    
    bn_freeze = args.bn_freeze
    if bn_freeze:
        modules = net.module.base_resnet.modules()
        for m in modules: 
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()
    
    

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):
        
        labels = torch.cat((label1, label2), 0)

        input1 = Variable(input1.to(device))
        input2 = Variable(input2.to(device))
        
        labels = Variable(labels.to(device))
        data_time.update(time.time() - end)

        feat, out0, = net(input1, input2)
        
        if args.proxy:
            
            loss_rgb = criterion_proxy_rgb(out0[:32][:], labels[:32][:])
            loss_ir = criterion_proxy_ir(out0[32:][:], labels[32:][:])
            loss = loss_rgb + loss_ir
            
        else:
            loss_id = criterion_id(out0, labels)
            loss_tri, batch_acc = criterion_tri(feat, labels)
            correct += (batch_acc / 2)
            _, predicted = out0.max(1)
            correct += (predicted.eq(labels).sum().item() / 2)
            
            loss = loss_id + loss_tri
            
        #loss_p2p, loss_p2e = criterion_two(out0, labels)
        
        #loss = loss_id + loss_tri
        #loss = loss_p2p + loss_p2e
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(net.module.parameters(), 10)
        if args.proxy:
            torch.nn.utils.clip_grad_value_(criterion_proxy_rgb.parameters(), 10)
            torch.nn.utils.clip_grad_value_(criterion_proxy_ir.parameters(), 10)
        
        optimizer.step()

        # update P
        if args.proxy:
            train_loss.update(loss.item(), 2 * input1.size(0))
        else:
            train_loss.update(loss.item(), 2 * input1.size(0))
            id_loss.update(loss_id.item(), 2 * input1.size(0))
            tri_loss.update(loss_tri.item(), 2 * input1.size(0))

        #p2p_loss.update(loss_p2p.item(), 3 * input1.size(0))
        #p2e_loss.update(loss_p2e.item(), 3 * input1.size(0))
        
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.5f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'p2eLoss: {p2e_loss.val:.4f} ({p2e_loss.avg:.4f}) '
                  'p2pLoss: {p2p_loss.val:.4f} ({p2p_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss, p2e_loss=p2e_loss, p2p_loss=p2p_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
    scheduler.step()
    #writer.add_scalar('p2p_loss', p2p_loss.avg, epoch)
    #writer.add_scalar('p2e_loss', p2e_loss.avg, epoch)
    
    #memory deallocation
    del input1, input2, labels, loss


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 512))
    gall_feat_att = np.zeros((ngall, 512))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.to(device))
            feat, feat_att = net(x1=input, modal=test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 512))
    query_feat_att = np.zeros((nquery, 512))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.to(device))
            feat, feat_att = net(x2=input, modal=test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP      = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)        
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    writer.add_scalar('rank1_att', cmc_att[0], epoch)
    writer.add_scalar('mAP_att', mAP_att, epoch)
    writer.add_scalar('mINP_att', mINP_att, epoch)
    
    # memory deallocation
    del input, label
    
    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


# training
print('==> Start Training...')
for epoch in range(start_epoch, 81 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    if args.warm > 0:
        
        unfreeze_model_param = list(net.module.base_resnet.model.embedding.parameters()) + list(criterion_proxy_ir.parameters()) + list(criterion_proxy_rgb.parameters())

        if epoch == 0:
            for param in list(set(net.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = False
        if epoch == args.warm:
            for param in list(set(net.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = True

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)
    # trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
    #                               num_workers=args.workers, drop_last=True)
    # training
    train(epoch)

    if epoch > 0 and epoch % 2 == 1:
        time.sleep(10)
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
        print(cmc)
        # save model
        if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_att[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc_att,
                'mAP': mAP_att,
                'mINP': mINP_att,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        # save model
        #if epoch > 10 and epoch % args.save_epoch == 0:
        state = {
            'net': net.state_dict(),
            'cmc': cmc,
            'mAP': mAP,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_path + suffix + '_current.t'.format(epoch))

        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('Best Epoch [{}]'.format(best_epoch))