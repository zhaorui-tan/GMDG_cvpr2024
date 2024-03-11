"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch
import wandb
from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer
import time
import torchvision.utils as vutils
import torch.nn.functional as F
from network.mynn import freeze_weights, unfreeze_weights
import numpy as np
import random
import torch.nn as nn

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', nargs='*', type=str, default=['cityscapes'],
                    help='a list of datasets; cityscapes, mapillary, camvid, kitti, gtav, mapillary, synthia')
parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                    help='uniformly sample images across the multiple source domains')
parser.add_argument('--val_dataset', nargs='*', type=str, default=['bdd100k'],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--covstat_val_dataset', nargs='*', type=str, default=['cityscapes'],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')
parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.0, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
# parser.add_argument('--syncbn', action='store_true', default=True,
#                     help='Use Synchronized BN')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
# parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
#                     help='url used to set up distributed training')

parser.add_argument('--wt_layer', nargs='*', type=int, default=[0,0,0,0,0,0,0],
                    help='0: None, 1: IW/IRW, 2: ISW, 3: IS, 4: IN (IBNNet: 0 0 4 4 4 0 0)')
parser.add_argument('--wt_reg_weight', type=float, default=0.0)
parser.add_argument('--relax_denom', type=float, default=2.0)
parser.add_argument('--clusters', type=int, default=50)
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--dynamic', action='store_true', default=False)

parser.add_argument('--image_in', action='store_true', default=False,
                    help='Input Image Instance Norm')
parser.add_argument('--cov_stat_epoch', type=int, default=5,
                    help='cov_stat_epoch')
parser.add_argument('--visualize_feature', action='store_true', default=False,
                    help='Visualize intermediate feature')
parser.add_argument('--use_wtloss', action='store_true', default=False,
                    help='Automatic setting from wt_layer')
parser.add_argument('--use_isw', action='store_true', default=False,
                    help='Automatic setting from wt_layer')
parser.add_argument('--wandb_name', type=str, default='',
                    help='use wandb and wandb name')

args = parser.parse_args()
N_DOMAINS = 2
# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True

# random_seed = cfg.RANDOM_SEED  #304
random_seed = 304  #304
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

# if 'WORLD_SIZE' in os.environ:
#     # args.apex = int(os.environ['WORLD_SIZE']) > 1
#     args.world_size = int(os.environ['WORLD_SIZE'])
#     print("Total world size: ", int(os.environ['WORLD_SIZE']))

# torch.cuda.set_device(args.local_rank)
# print('My Rank:', args.local_rank)
# # Initialize distributed communication
# args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)
#
# torch.distributed.init_process_group(backend='nccl',
#                                      init_method=args.dist_url,
#                                      world_size=args.world_size,
#                                      rank=args.local_rank)

for i in range(len(args.wt_layer)):
    if args.wt_layer[i] == 1:
        args.use_wtloss = True
    if args.wt_layer[i] == 2:
        args.use_wtloss = True
        args.use_isw = True



def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def sample_covariance(a, b, invert=False):
    '''
    Sample covariance estimating
    a = [N,m]
    b = [N,m]
    '''
    # assert (a.shape[0] == b.shape[0])
    # assert (a.shape[1] == b.shape[1])
    # m = a.shape[1]
    N = a.shape[0]
    C = torch.matmul(a.T, b)/ N
    if invert:
        return torch.linalg.pinv(C)
    else:
        return C

def get_cond_shift(X1, Y1, estimator=sample_covariance):
    m1 = torch.mean(X1, dim=0)
    my1 = torch.mean(Y1, dim=0)
    x1 = X1 - m1
    y1 = Y1 - my1

    shift_loss = 0.
    for i in range(len(x1)):
        c_x1_y = estimator(x1[i], y1[i])
        c_y_x1 = estimator(y1[i], x1[i])
        inv_c_y_y = estimator(y1[i], y1[i], invert=True)
        shift = torch.matmul(c_x1_y, torch.matmul(inv_c_y_y, c_y_x1))
        shift_loss += nn.MSELoss()(shift, torch.zeros_like(shift))
    return shift_loss


class MeanEncoder(nn.Module):
    """Identity function"""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x


class VarianceEncoder(nn.Module):
    """Bias-only model with diagonal covariance"""
    def __init__(self, shape, init=0.1, channelwise=True, eps=1e-5):
        super().__init__()
        self.shape = shape
        self.eps = eps

        init = (torch.as_tensor(init - eps).exp() - 1.0).log()
        b_shape = shape
        if channelwise:
            if len(shape) == 4:
                # [B, C, H, W]
                # b_shape = (1, shape[1], 1, 1)
                b_shape = (1, shape[1], shape[2], shape[3])
            elif len(shape ) == 3:
                # CLIP-ViT: [H*W+1, B, C]
                b_shape = (1, 1, shape[2])
            elif len(shape) == 2:
                # CLIP-ViT: [B, C]
                b_shape = (1, shape[1])
            else:
                raise ValueError()

        self.b = nn.Parameter(torch.full(b_shape, init))

    def forward(self, x):
        return F.softplus(self.b) + self.eps


def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)
    if args.wandb_name:
        # if args.local_rank == 0:
         wandb.init(project='Res_DG3', name=args.wandb_name, config=args)

    train_loader, val_loaders, train_obj, extra_val_loaders, covstat_val_loaders = datasets.setup_loaders(args)

    criterion, criterion_val = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    net = network.get_net(args, criterion, criterion_aux)
    net.cuda()
    net = nn.DataParallel(net)
    optim, scheduler = optimizer.get_optimizer(args, net)

    oracle_model = network.get_net(args, criterion, criterion_aux)
    for name, para in oracle_model.named_parameters():
        para.requires_grad = False
    oracle_model.cuda()
    oracle_model = nn.DataParallel(oracle_model)

    Y_args = args
    Y_args.use_wtloss = False
    Y_net = network.get_net(Y_args, criterion, criterion_aux)
    Y_net.cuda()
    Y_net = nn.DataParallel(Y_net)
    Y_optim, Y_scheduler = optimizer.get_optimizer(Y_args, Y_net)

    f_mean_encoders = MeanEncoder((1, 19 , 1, 1))
    f_var_encoders =VarianceEncoder((1, 19, 1, 1))
    f_mean_encoders = f_mean_encoders.cuda()
    f_var_encoders = f_var_encoders.cuda()
    f_mean_encoders = nn.DataParallel(f_mean_encoders, )
    f_var_encoders = nn.DataParallel(f_var_encoders, )
    f_var_optim, f_var_scheduler = optimizer.get_optimizer(Y_args, f_var_encoders)


    d_mean_encoders = nn.ModuleList([MeanEncoder((1, 19 * 2, 1, 1)) for _ in range(N_DOMAINS)])
    d_var_encoders = nn.ModuleList([VarianceEncoder((1, 19 * 2, 1, 1)) for _ in range(N_DOMAINS)])
    d_mean_encoders = d_mean_encoders.cuda()
    d_var_encoders = d_var_encoders.cuda()
    d_mean_encoders = nn.DataParallel(d_mean_encoders, )
    d_var_encoders = nn.DataParallel(d_var_encoders, )
    d_var_optim, d_var_scheduler = optimizer.get_optimizer(Y_args, d_var_encoders)

    epoch = 0
    i = 0

    if args.snapshot:
        epoch, mean_iu = optimizer.load_weights(net, optim, scheduler,
                            args.snapshot, args.restore_optimizer)
        if args.restore_optimizer is True:
            iter_per_epoch = len(train_loader)
            i = iter_per_epoch * epoch
        else:
            epoch = 0

    print("#### iteration", i)
    torch.cuda.empty_cache()
    # Main Loop
    # for epoch in range(args.start_epoch, args.max_epoch):

    while i < args.max_iter:
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.ITER = i
        cfg.immutable(True)

        i = train(train_loader, net, Y_net, optim,  Y_optim,  epoch, writer, scheduler, Y_scheduler, args.max_iter, extra_val_loaders, criterion_val,
                  d_mean_encoders, d_var_encoders, d_var_optim, d_var_scheduler,
                  oracle_model, f_mean_encoders, f_var_encoders, f_var_optim, f_var_scheduler, )
        # train_loader.sampler.set_epoch(epoch + 1)

        if (args.dynamic and args.use_isw and epoch % (args.cov_stat_epoch + 1) == args.cov_stat_epoch) \
           or (args.dynamic is False and args.use_isw and epoch == args.cov_stat_epoch):
            net.module.reset_mask_matrix()
            for trial in range(args.trials):
                for dataset, val_loader in covstat_val_loaders.items():  # For get the statistics of covariance
                    validate_for_cov_stat(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i,
                                          save_pth=False)
                    net.module.set_mask_matrix()

        print("Saving pth file...")
        evaluate_eval(args, net, optim, scheduler, None, None, [],
                    writer, epoch, "None", None, i, save_pth=True)

        # if args.class_uniform_pct:
        #     if epoch >= args.max_cu_epoch:
        #         train_obj.build_epoch(cut=True)
        #         train_loader.sampler.set_num_samples()
        #     else:
        #         train_obj.build_epoch()
        for dataset, val_loader in extra_val_loaders.items():
            # TODO
            torch.cuda.empty_cache()
            print("Extra validating... This won't save pth file")
            with torch.no_grad():
                mean_iu = validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, args.max_iter, save_pth=False)
                eval_res = {
                        dataset:
                        {'cur_miou': mean_iu,
                         'epoch': epoch,}}
                if args.wandb_name:
                    wandb.log(eval_res)
        epoch += 1

    # Validation after epochs
    if len(val_loaders) == 1:
        # Run validation only one time - To save models
        for dataset, val_loader in val_loaders.items():
            validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i)
    else:
        # if args.local_rank == 0:
        print("Saving pth file...")
        evaluate_eval(args, net, optim, scheduler, None, None, [],
                    writer, epoch, "None", None, i, save_pth=True)

    for dataset, val_loader in extra_val_loaders.items():
        print("Extra validating... This won't save pth file")
        miou = validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=True)
        if args.wandb_name:
            wandb.log({
                dataset: miou
            })


def train(train_loader, net, Y_net, optim, Y_optim, curr_epoch, writer, scheduler,Y_scheduler, max_iter, extra_val_loaders, criterion_val,
          d_mean_encoders, d_var_encoders, d_var_optim, d_var_scheduler,
          oracle_model, f_mean_encoders, f_var_encoders, f_var_optim, f_var_scheduler,):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()
    Y_net.train()
    d_var_encoders.train()

    train_total_loss = AverageMeter()
    Y_train_total_loss = AverageMeter()
    d_shift_train_loss = AverageMeter()
    shift_train_loss = AverageMeter()
    vlb_train_loss = AverageMeter()

    time_meter = AverageMeter()

    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(train_loader):
        if curr_iter >= max_iter:
            break

        inputs, gts, _, aux_gts = data

        # Multi source and AGG case
        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            num_domains = D
            inputs = inputs.transpose(0, 1)
            gts = gts.transpose(0, 1).squeeze(2)
            aux_gts = aux_gts.transpose(0, 1).squeeze(2)

            inputs = [input.squeeze(0) for input in torch.chunk(inputs, num_domains, 0)]
            gts = [gt.squeeze(0) for gt in torch.chunk(gts, num_domains, 0)]
            aux_gts = [aux_gt.squeeze(0) for aux_gt in torch.chunk(aux_gts, num_domains, 0)]
        else:
            B, C, H, W = inputs.shape
            num_domains = 1
            inputs = [inputs]
            gts = [gts]
            aux_gts = [aux_gts]

        batch_pixel_size = C * H * W
        for di, ingredients in enumerate(zip(inputs, gts, aux_gts)):
            input, gt, aux_gt = ingredients

            start_ts = time.time()

            img_gt = None
            input, gt = input.cuda(), gt.cuda()

            gt_ = gt.squeeze(0).repeat(3,1,1,1).permute(1,0,2,3).float().cuda('cuda:1')
            Y_outputs = Y_net(gt_, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=args.visualize_feature)
            Y_features_ori = Y_outputs['aux_out']  # [8, 1024, 48, 48]

            outputs_index = 0
            Y_main_loss = Y_outputs['return_loss'][outputs_index]
            outputs_index += 1
            Y_aux_loss = Y_outputs['return_loss'][outputs_index]
            outputs_index += 1

            Y_total_loss = Y_main_loss + (0.4 * Y_aux_loss)
            Y_total_loss = Y_total_loss.mean()


            Y_optim.zero_grad()
            Y_total_loss.backward()
            Y_optim.step()
            log_Y_loss = Y_total_loss.clone().detach_()
            Y_train_total_loss.update(log_Y_loss.item(), batch_pixel_size)


            if args.use_isw:
                outputs = net(input, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=args.visualize_feature,
                            apply_wtloss=False if curr_epoch<=args.cov_stat_epoch else True)
            else:
                outputs = net(input, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=args.visualize_feature)

            outputs_index = 0
            main_loss = outputs['return_loss'][outputs_index]

            outputs_index += 1
            aux_loss = outputs['return_loss'][outputs_index]
            outputs_index += 1
            total_loss = main_loss + (0.4 * aux_loss)

            if args.use_wtloss and (not args.use_isw or (args.use_isw and curr_epoch > args.cov_stat_epoch)):
                wt_loss = outputs['return_loss'][outputs_index]
                outputs_index += 1
                total_loss = total_loss + (args.wt_reg_weight * wt_loss)
            else:
                wt_loss = 0

            if args.visualize_feature:
                f_cor_arr = outputs['return_loss'][outputs_index]
                outputs_index += 1

            total_loss = total_loss.mean()
            optim.zero_grad()
            total_loss.backward()
            optim.step()

            log_total_loss = total_loss.clone().detach_()
            train_total_loss.update(log_total_loss.item(), batch_pixel_size)


            reg_loss = 0.
            outputs = net(input, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=args.visualize_feature)
            features = outputs['aux_out']  # [8, 1024, 48, 48]
            d_reg = 0.
            d_all_means = []
            d_all_vars = []
            shape = features.shape
            Y_features = Y_features_ori.detach()


            features_ = features.view(2, shape[0] // 2, shape[1], shape[2], shape[3])
            Y_features_ = Y_features.view(2, shape[0] // 2, shape[1], shape[2], shape[3])
            for dd in range(len(features_)):
                x_y_feature = torch.cat([features_[dd], Y_features_[dd]], dim=1)
                d_mean = d_mean_encoders.module[dd](x_y_feature)
                d_var = d_var_encoders.module[dd](x_y_feature)
                d_all_means.append(d_mean)
                d_all_vars.append(d_var)

            d_all_means_mean = torch.stack(d_all_means).mean(0).mean(0).unsqueeze(0).cuda('cuda:1')
            d_all_vars_mean = torch.stack(d_all_vars).mean(0).cuda('cuda:1')
            feat_y = torch.cat([features, Y_features], dim=1).cuda('cuda:1')
            vlb = (d_all_means_mean
                   - feat_y.detach()).pow(2).div(d_all_vars_mean) + d_all_vars_mean.log()

            d_reg += vlb.mean() / 2.
            d_reg = d_reg * 0.0001
            reg_loss += d_reg

            optim.zero_grad()
            d_var_optim.zero_grad()
            reg_loss.backward()
            optim.step()
            d_var_optim.step()

            log_d_reg = d_reg.clone().detach_()
            d_shift_train_loss.update(log_d_reg.item(), batch_pixel_size)


            del total_loss, log_total_loss,Y_total_loss, log_Y_loss, d_reg, log_d_reg


            outputs = net(input, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=args.visualize_feature)
            features = outputs['aux_out']  # [8, 1024, 48, 48]

            with torch.no_grad():
                oracle_outputs = oracle_model(input, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=args.visualize_feature)
                oracle_features = oracle_outputs['aux_out']


            mean = f_mean_encoders(features)
            var = f_var_encoders(features).mean(0)
            vlb = (mean - oracle_features).pow(2).div(var) + var.log()
            vlb = vlb.mean() * 0.0001

            optim.zero_grad()
            f_var_optim.zero_grad()
            vlb.backward()
            optim.step()
            f_var_optim.step()

            log_vlb = vlb.clone().detach_()
            vlb_train_loss.update(log_vlb.item(), batch_pixel_size)

            del vlb


            time_meter.update(time.time() - start_ts)

            if i % 50 == 49:
                if args.visualize_feature:
                    visualize_matrix(writer, f_cor_arr, curr_iter, '/Covariance/Feature-')

                msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                    curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss.avg,
                    optim.param_groups[-1]['lr'], time_meter.avg / args.train_batch_size)

                print(
                    f" iter {curr_iter}"
                    f" log_total_loss, {train_total_loss.avg:.4f}"
                    f" Y_main_loss, {Y_train_total_loss.avg:.4f}"
                    f" log_d_reg, {d_shift_train_loss.avg:.4f}"
                    f" vlb, {vlb_train_loss.avg:.4f}"
                )
                logging.info(msg)
                if args.use_wtloss:
                    print("Whitening Loss", wt_loss)
                if args.wandb_name:
                    wandb.log({
                        'Whitening Loss"': train_total_loss.avg,
                    })
                # Log tensorboard metrics for each iteration of the training phase
                writer.add_scalar('loss/train_loss', (train_total_loss.avg),
                                curr_iter)
                train_total_loss.reset()
                time_meter.reset()


        curr_iter += 1
        scheduler.step()
        Y_scheduler.step()
        d_var_scheduler.step()
        f_var_scheduler.step()

        if i > 5 and args.test_mode:
            return curr_iter

    return curr_iter

def validate(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    error_acc = 0
    dump_images = []
    with torch.no_grad():
        for val_idx, data in enumerate(val_loader):
            # input        = torch.Size([1, 3, 713, 713])
            # gt_image           = torch.Size([1, 713, 713])
            torch.cuda.empty_cache()
            inputs, gt_image, img_names, _ = data

            if len(inputs.shape) == 5:
                B, D, C, H, W = inputs.shape
                inputs = inputs.view(-1, C, H, W)
                gt_image = gt_image.view(-1, 1, H, W)

            assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
            assert inputs.size()[2:] == gt_image.size()[1:]

            batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
            inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

            with torch.no_grad():
                if args.use_wtloss:
                    output, f_cor_arr = net(inputs, visualize=True)
                else:
                    output = net(inputs)

            del inputs

            assert output.size()[2:] == gt_image.size()[1:]
            assert output.size()[1] == datasets.num_classes

            val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

            del gt_cuda

            # Collect data from different GPU to a single GPU since
            # encoding.parallel.criterionparallel function calculates distributed loss
            # functions
            predictions = output.data.max(1)[1].cpu()

            # Logging
            if val_idx % 20 == 0:
                # if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
            if val_idx > 10 and args.test_mode:
                break

            # Image Dumps
            if val_idx < 10:
                dump_images.append([gt_image, predictions, img_names])

            iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                                 datasets.num_classes)
            del output, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    # torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()

    # if args.local_rank == 0:
    mean_iu = evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)

    if args.use_wtloss:
        visualize_matrix(writer, f_cor_arr, curr_iter, '/Covariance/Feature-')

    # return val_loss.avg
    return mean_iu

def validate_for_cov_stat(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    # net.train()#eval()
    net.eval()

    for val_idx, data in enumerate(val_loader):
        img_or, img_photometric, img_geometric, img_name = data   # img_geometric is not used.
        img_or, img_photometric = img_or.cuda(), img_photometric.cuda()

        with torch.no_grad():
            net(x=[img_photometric, img_or], cal_covstat=True)

        del img_or, img_photometric, img_geometric

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / 100", val_idx + 1)
        del data

        if val_idx >= 499:
            return


def visualize_matrix(writer, matrix_arr, iteration, title_str):
    stage = 'valid'

    for i in range(len(matrix_arr)):
        C = matrix_arr[i].shape[1]
        matrix = matrix_arr[i][0].unsqueeze(0)    # 1 X C X C
        matrix = torch.clamp(torch.abs(matrix), max=1)
        matrix = torch.cat((torch.ones(1, C, C).cuda(), torch.abs(matrix - 1.0),
                        torch.abs(matrix - 1.0)), 0)
        matrix = vutils.make_grid(matrix, padding=5, normalize=False, range=(0,1))
        writer.add_image(stage + title_str + str(i), matrix, iteration)


def save_feature_numpy(feature_maps, iteration):
    file_fullpath = '/home/userA/projects/visualization/feature_map/'
    file_name = str(args.date) + '_' + str(args.exp)
    B, C, H, W = feature_maps.shape
    for i in range(B):
        feature_map = feature_maps[i]
        feature_map = feature_map.data.cpu().numpy()   # H X D
        file_name_post = '_' + str(iteration * B + i)
        np.save(file_fullpath + file_name + file_name_post, feature_map)


if __name__ == '__main__':
    main()
