# Copyright (c) Kakao Brain. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.optimizers import get_optimizer
from domainbed.networks.ur_networks import URFeaturizer
from domainbed.lib import misc
from domainbed.algorithms import Algorithm


class ForwardModel(nn.Module):
    """Forward model is used to reduce gpu memory usage of SWAD.
    """
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.predict(x)

    def predict(self, x):
        return self.network(x)


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
                b_shape = (1, shape[1], 1, 1)
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


class YEncoder(nn.Module):
    """Identity function"""
    def __init__(self, inshape, outshape):
        super().__init__()
        self.inshape = inshape
        self.outshape = outshape
        self.l1 = nn.Linear(self.inshape, self.outshape//2)
        self.l2 = nn.Linear(self.outshape//2, self.outshape)
        self.l3 = nn.Linear(self.outshape, self.outshape)
        self.norm = nn.LayerNorm(outshape)

    def forward(self, x):
        x = self.l1(x)
        x = nn.Dropout(p=0.2)(x)
        x = nn.ReLU()(x)
        x = self.l2(x)
        x = nn.Dropout(p=0.2)(x)
        x = nn.ReLU()(x)
        x = self.l3(x)
        x = nn.Dropout(p=0.2)(x)

        n = torch.randn_like(x).cuda()
        scale = x.detach().abs().max().item() / (n.abs().max().item() + 1e-6)
        n = n * scale * 0.1
        xn = x + n
        return x, xn


def get_shapes(model, input_shape):
    # get shape of intermediate features
    with torch.no_grad():
        dummy = torch.rand(1, *input_shape).to(next(model.parameters()).device)
        _, feats = model(dummy, ret_feats=True)
        shapes = [f.shape for f in feats]

    return shapes


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
    # print(matrix1.shape, matrix2.shape)
    m1 = torch.mean(X1, dim=0)
    my1 = torch.mean(Y1, dim=0)
    x1 = X1 - m1
    y1 = Y1 - my1

    c_x1_y = estimator(x1, y1)
    c_y_x1 = estimator(y1, x1)

    inv_c_y_y = estimator(y1, y1, invert=True)
    shift = torch.matmul(c_x1_y, torch.matmul(inv_c_y_y, c_y_x1))
    return nn.MSELoss()(shift, torch.zeros_like(shift))


def get_rss(h1, h2):
    h1, h2 = h1.float(), h2.float()
    h1t = h1.permute(1, 0)
    if h1.shape[0] >= h1.shape[1]:
        pinv = torch.matmul(torch.linalg.pinv(torch.matmul(h1t, h1)), h1t)
    else:
        pinv = torch.matmul(h1t,torch.linalg.pinv(torch.matmul(h1, h1t)))
    K = torch.matmul(pinv, h2)
    rss = 0
    h1_ = nn.functional.linear(h1,K)
    rss += nn.MSELoss()(h1_, h2)
    return rss

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class GMDG(Algorithm):
    """Mutual-Information Regularization with Oracle"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.pre_featurizer = URFeaturizer(
            input_shape, self.hparams, freeze="all", feat_layers=hparams.feat_layers
        )
        self.featurizer = URFeaturizer(
            input_shape, self.hparams, feat_layers=hparams.feat_layers
        )
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.ld = hparams.ld

        self.shift = hparams.shift
        self.d_shift = hparams.d_shift
        self.seen_domains = num_domains
        self.use_MIRO = hparams.use_MIRO
        self.use_condition1 = hparams.use_condition1
        self.use_condition2 = hparams.use_condition2

        # build mean/var encoders
        shapes = get_shapes(self.pre_featurizer, self.input_shape)
        self.mean_encoders = nn.ModuleList([
            MeanEncoder(shape) for shape in shapes
        ])
        self.var_encoders = nn.ModuleList([
            VarianceEncoder(shape) for shape in shapes
        ])

        self.d_mean_encoders =  nn.ModuleList([
            MeanEncoder(shape=[1,self.featurizer.n_outputs * 2]) for _ in range(self.seen_domains)
        ])
        self.d_var_encoders = nn.ModuleList([
            VarianceEncoder(shape=[1, self.featurizer.n_outputs * 2]) for _ in range(self.seen_domains)
        ])
        self.norm = nn.LayerNorm(self.featurizer.n_outputs)



        parameters = [
            {"params": self.network.parameters()},
            {"params": self.mean_encoders.parameters(), "lr": hparams.lr * hparams.lr_mult},
            {"params": self.var_encoders.parameters(), "lr": hparams.lr * hparams.lr_mult},
            {"params": self.d_mean_encoders.parameters(), "lr": hparams.lr * hparams.lr_mult},
            {"params": self.d_var_encoders.parameters(), "lr": hparams.lr * hparams.lr_mult},
            {"params": self.norm.parameters(), "lr": hparams.lr * hparams.lr_mult}
        ]

        self.optimizer = get_optimizer(
            hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.y_mapping = YEncoder(self.featurizer.n_outputs, self.featurizer.n_outputs)
        y_parameters = [{"params": self.y_mapping.parameters(), "lr": hparams.lr * hparams.lr_mult},
                        ]
        self.y_optimizer = get_optimizer(
            hparams["optimizer"],
            y_parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, all_x, all_y, feat, inter_feats, loss, gd_loss_0, step_ratio):
        all_loss = 0.
        all_loss += loss
        returns = {}

        returns['loss'] = loss.item()

        y = all_y.unsqueeze(1).repeat(1, self.featurizer.n_outputs) * 1.
        y_, _ = self.y_mapping(y)
        y_class_logit = self.classifier(y_)
        y_loss = F.cross_entropy(y_class_logit, all_y)
        self.y_optimizer.zero_grad()
        y_loss.backward()
        self.y_optimizer.step()

        returns['y_loss'] = y_loss.item()

        # MIRO, GReg1
        with torch.no_grad():
            _, pre_feats = self.pre_featurizer(all_x, ret_feats=True)

        reg_loss = 0.
        if self.use_MIRO:
            for f, pre_f, mean_enc, var_enc in misc.zip_strict(
                inter_feats, pre_feats, self.mean_encoders, self.var_encoders
            ):
                # mutual information regularization
                mean = mean_enc(f)
                var = var_enc(f)
                vlb = (mean - pre_f).pow(2).div(var) + var.log()
                reg_loss += vlb.mean() / 2.
            reg_loss = reg_loss * self.ld
            all_loss += reg_loss
            returns['reg_loss'] = reg_loss.item()


        if self.shift > 0:
            if self.use_condition1: # GAim1
                feat_norm = l2normalize(feat)
                y_norm = l2normalize(y_)
                x_shift = get_cond_shift(feat_norm, y_norm.detach(), )
                gd_shift_0 = torch.autograd.grad(outputs=x_shift, inputs=feat, retain_graph=True)
                scale = gd_loss_0 / gd_shift_0[0].max()
                x_shift = x_shift * scale * self.shift
                all_loss += x_shift
                returns['x_shift'] = x_shift.item()
            else: # iAim1
                feat_norm = l2normalize(feat)
                shape = feat.shape
                f_ = feat_norm.view(self.seen_domains, shape[0] // self.seen_domains, shape[1], )
                cov_list = []
                for ff in f_:
                    cov = sample_covariance(ff,ff)
                    cov_list.append(cov)
                cov_list = torch.stack(cov_list)
                cov_mean = cov_list.mean(0)
                x_shift = F.mse_loss(cov_list, cov_mean)
                gd_shift_0 = torch.autograd.grad(outputs=x_shift, inputs=feat, retain_graph=True)
                scale = gd_loss_0 / gd_shift_0[0].max()
                x_shift = x_shift * scale * self.shift
                all_loss += x_shift
                returns['x_shift'] = x_shift.item()

        if self.d_shift > 0:
            if self.use_condition2: # GReg2
                d_reg = 0.
                shape = feat.shape
                f_ = feat.view(self.seen_domains, shape[0]//self.seen_domains, shape[1],)
                y_ = y_.detach().view(self.seen_domains, shape[0]//self.seen_domains, shape[1],)
                d_all_means = []
                d_all_vars = []
                for i in range(self.seen_domains):
                    d_mean = self.d_mean_encoders[i](torch.cat([f_[i], y_[i]], dim=-1))
                    d_var = self.d_var_encoders[i](torch.cat([f_[i], y_[i]], dim=-1))
                    d_all_means.append(d_mean)
                    d_all_vars.append(d_var)

                d_all_means_mean = torch.stack(d_all_means).mean(0)
                d_all_vars_mean = torch.stack(d_all_vars).mean(0)
                feat_y = torch.cat([f_, y_], dim=-1)
                vlb = (d_all_means_mean
                       - feat_y.detach()).pow(2).div(d_all_vars_mean) + d_all_vars_mean.log()
                d_reg += vlb.mean() / 2.
                d_reg = d_reg * self.d_shift
                all_loss += d_reg
                returns['d_reg'] = d_reg.item()

            else: # iReg2
                d_reg = 0.
                shape = feat.shape
                f_ = feat.view(self.seen_domains, shape[0] // self.seen_domains, shape[1], )
                d_all_means = []
                d_all_vars = []
                for i in range(self.seen_domains):
                    d_mean = self.d_mean_encoders[i](torch.cat([f_[i], f_[i]], dim=-1))
                    d_var = self.d_var_encoders[i](torch.cat([f_[i], f_[i]], dim=-1))
                    d_all_means.append(d_mean)
                    d_all_vars.append(d_var)

                # TODO
                d_all_means_mean = torch.stack(d_all_means).mean(0)
                d_all_vars_mean = torch.stack(d_all_vars).mean(0)
                feat_y = torch.cat([f_, f_], dim=-1)
                vlb = (d_all_means_mean
                       - feat_y.detach()).pow(2).div(d_all_vars_mean) + d_all_vars_mean.log()
                d_reg += vlb.mean() / 2.
                d_reg = d_reg * self.d_shift
                all_loss += d_reg
                returns['d_reg'] = d_reg.item()

        self.optimizer.zero_grad()
        all_loss.backward()
        self.optimizer.step()
        return returns

    def forward(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        feat, inter_feats = self.featurizer(all_x, ret_feats=True)
        logit = self.classifier(feat)
        loss = F.cross_entropy(logit, all_y)  # GAim2
        gd_loss = torch.autograd.grad(outputs=loss, inputs=feat, retain_graph=True)
        return all_x, all_y, feat, inter_feats, loss, gd_loss[0].max()

    def predict(self, x):
        return self.network(x)

    def get_forward_model(self):
        forward_model = ForwardModel(self.network)
        return forward_model