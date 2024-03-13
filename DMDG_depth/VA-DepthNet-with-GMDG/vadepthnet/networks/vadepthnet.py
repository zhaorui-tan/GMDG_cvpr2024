import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .loss import VarLoss, SILogLoss
########################################################################################################################
from .basic_res import BasicBlock
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
            elif len(shape) == 3:
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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=4),
            # nn.BatchNorm2d(mid_channels),
            nn.InstanceNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            # ModulatedDeformConvPack(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(),
        )

        self.bt = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        skip = self.bt(x)

        x = self.channel_shuffle(x, 4)

        x = self.conv1(x)

        x = self.conv2(x)

        return x + skip

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.shape

        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(
            in_channels, out_channels, in_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            if diffX > 0 or diffY > 0:
                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, prior_mean=1.54):
        super(OutConv, self).__init__()

        self.prior_mean = prior_mean
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.exp(self.conv(x) + self.prior_mean)


class VarLayer(nn.Module):
    def __init__(self, in_channels, h, w):
        super(VarLayer, self).__init__()

        self.gr = 16

        self.grad = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, 4 * self.gr, kernel_size=3, padding=1))

        self.att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, 4 * self.gr, kernel_size=3, padding=1),
            nn.Sigmoid())

        num = h * w

        a = torch.zeros(num, 4, num, dtype=torch.float16)

        for i in range(num):

            # a[i, 0, i] = 1.0
            # if i + 1 < num:
            if (i + 1) % w != 0 and (i + 1) < num:
                a[i, 0, i] = 1.0
                a[i, 0, i + 1] = -1.0

            # a[i, 1, i] = 1.0
            if i + w < num:
                a[i, 1, i] = 1.0
                a[i, 1, i + w] = -1.0

            if (i + 2) % w != 0 and (i + 2) < num:
                a[i, 2, i] = 1.0
                a[i, 2, i + 2] = -1.0

            if i + w + w < num:
                a[i, 3, i] = 1.0
                a[i, 3, i + w + w] = -1.0

        a[-1, 0, -1] = 1.0
        a[-1, 1, -1] = 1.0

        a[-1, 2, -1] = 1.0
        a[-1, 3, -1] = 1.0

        self.register_buffer('a', a.unsqueeze(0))

        self.ins = nn.GroupNorm(1, self.gr)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels // 2, self.gr, kernel_size=1, padding=0),
            nn.Sigmoid())

        self.post = nn.Sequential(
            nn.Conv2d(self.gr, 8 * self.gr, kernel_size=3, padding=1))

    def forward(self, x):
        skip = x.clone()
        att = self.att(x)
        grad = self.grad(x)

        se = self.se(x)

        n, c, h, w = x.shape

        att = att.reshape(n * self.gr, 4, h * w, 1).permute(0, 2, 1, 3)
        grad = grad.reshape(n * self.gr, 4, h * w, 1).permute(0, 2, 1, 3)

        A = self.a * att
        B = grad * att

        A = A.reshape(n * self.gr, h * w * 4, h * w)
        B = B.reshape(n * self.gr, h * w * 4, 1)

        AT = A.permute(0, 2, 1)

        ATA = torch.bmm(AT, A)
        ATB = torch.bmm(AT, B)

        jitter = torch.eye(n=h * w, dtype=x.dtype, device=x.device).unsqueeze(0) * 1e-12
        x, _ = torch.solve(ATB, ATA + jitter)

        x = x.reshape(n, self.gr, h, w)

        x = self.ins(x)

        x = se * x

        x = self.post(x)

        return x


class Refine(nn.Module):
    def __init__(self, c1, c2):
        super(Refine, self).__init__()

        s = c1 + c2
        self.fw = nn.Sequential(
            nn.Conv2d(s, s, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(s, c1, kernel_size=3, padding=1))

        self.dw = nn.Sequential(
            nn.Conv2d(s, s, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(s, c2, kernel_size=3, padding=1))

    def forward(self, feat, depth):
        cc = torch.cat([feat, depth], 1)
        feat_new = self.fw(cc)
        depth_new = self.dw(cc)
        return feat_new, depth_new


class MetricLayer(nn.Module):
    def __init__(self, c):
        super(MetricLayer, self).__init__()

        self.ln = nn.Sequential(
            nn.Linear(c, c // 4),
            nn.LeakyReLU(),
            nn.Linear(c // 4, 2))

    def forward(self, x):
        x = x.squeeze(-1).squeeze(-1)
        x = self.ln(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        return x

def sample_covariance(a, b, invert=False):
    '''
    Sample covariance estimating
    a = [N,m]
    b = [N,m]
    '''
    N = a.shape[0]
    C = torch.matmul(a.T, b)/ N
    if invert:
        return torch.pinverse(C)
    else:
        return C

def get_cond_shift(X1, Y1, estimator=sample_covariance):
    # print(matrix1.shape, matrix2.shape)
    m1 = torch.mean(X1, dim=0)
    my1 = torch.mean(Y1, dim=0)
    x1 = X1 - m1
    y1 = Y1 - my1

    # torch.Size([3, 1536, 15, 20])
    c_x1_y = estimator(x1, y1)
    c_y_x1 = estimator(y1, x1)

    inv_c_y_y = estimator(y1, y1, invert=True)
    shift = torch.matmul(c_x1_y, torch.matmul(inv_c_y_y, c_y_x1))
    return nn.MSELoss()(shift, torch.zeros_like(shift))


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class VADepthNet(nn.Module):
    def __init__(self, pretrained=None, max_depth=10.0, prior_mean=1.54, si_lambda=0.85, img_size=(480, 640), args = None):
        super().__init__()

        self.prior_mean = prior_mean
        self.SI_loss_lambda = si_lambda
        self.max_depth = max_depth

        pretrain_img_size = img_size
        patch_size = (4, 4)
        in_chans = 3
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
        window_size = 12

        backbone_cfg = dict(
            pretrain_img_size=pretrain_img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=True,
            drop_rate=0.,
            # use_checkpoint=True
        )

        self.backbone = SwinTransformer(**backbone_cfg)
        self.backbone_fixed = SwinTransformer(**backbone_cfg)
        for n, p in self.backbone_fixed.named_parameters():
            p.requires_grad = False

        self.backbone.init_weights(pretrained=pretrained)

        self.up_4 = Up(1536 + 768, 512)
        self.up_3 = Up(512 + 384, 256)
        self.up_2 = Up(256 + 192, 64)

        self.outc = OutConv(128, 1, self.prior_mean)

        self.vlayer = VarLayer(512, img_size[0] // 16, img_size[1] // 16)

        self.ref_4 = Refine(512, 128)
        self.ref_3 = Refine(256, 128)
        self.ref_2 = Refine(64, 128)

        self.var_loss = VarLoss(128, 512)
        self.si_loss = SILogLoss(self.SI_loss_lambda, self.max_depth)

        self.mlayer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            MetricLayer(1536))

        if args.use_o:
            if args.te_idx == -1:
                shape = [4, 1536, 15, 20]
            else:
                shape = [3, 1536, 15, 20]
            self.mean_encoder = MeanEncoder(shape = shape)
            self.var_encoder = VarianceEncoder(shape = shape)


        if args.use_y:
            if args.te_idx == -1:
                shape = [4, 1536 * 2, 15, 20]
                self.y_mapping = BasicBlock(1536, 1536)
                self.d_mean_encoders = nn.ModuleList([
                    MeanEncoder(shape=shape) for _ in range(4)
                ])
                self.d_var_encoders = nn.ModuleList([
                    VarianceEncoder(shape=shape) for _ in range(4)
                ])

            else:
                shape = [3, 1536 * 2, 15, 20]
                self.y_mapping = BasicBlock(1536, 1536)
                self.d_mean_encoders = nn.ModuleList([
                    MeanEncoder(shape=shape) for _ in range(3)
                ])
                self.d_var_encoders = nn.ModuleList([
                    VarianceEncoder(shape=shape) for _ in range(3)
                ])



    def forward(self, x, gts=None, optimizer=None, optimizer_y=None, y_mapping=None, args=None):
        torch.cuda.empty_cache()
        if self.training:
            if args.use_y:
                y_ = gts.permute(1, 0, 2, 3).repeat(3, 1, 1, 1).permute(1, 0, 2, 3)
                with torch.no_grad():
                    y2, y3, y4, y5 = self.backbone_fixed(y_)
                y5 = self.y_mapping(y5)

                outs, loss = self._forward_helper(y2, y3, y4, y5, gts)
                loss = loss * 0.01
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                del y2, y3, y4, y_
                torch.cuda.empty_cache()

            x2, x3, x4, x5 = self.backbone(x)
            reg_loss = 0.0
            if args.use_o:
                with torch.no_grad():
                    fixed_x2, fixed_x3, fixed_x4, fixed_x5 = self.backbone_fixed(x)
                    del fixed_x2, fixed_x3, fixed_x4
                mean = self.mean_encoder(x5)
                var = self.var_encoder(x5)
                vlb = (mean - fixed_x5).pow(2).div(var) + var.log()
                reg_loss += vlb.mean() / 2. * 0.001

            d_reg = 0.
            if args.use_y:
                x_y = torch.cat((x5, y5.detach()), dim=1)
                if args.te_idx == -1:
                    x_y = x_y.view(4, -1, 3072, 15, 20)
                else:
                    x_y = x_y.view(3, -1, 3072, 15, 20)
                d_all_means = []
                d_all_vars = []

                if args.te_idx == -1:
                    for i in range(4):
                        mean = self.d_mean_encoders[i](x_y[i])
                        var = self.d_var_encoders[i](x_y[i])
                        d_all_means.append(mean)
                        d_all_vars.append(var)
                else:
                    for i in range(3):
                        mean = self.d_mean_encoders[i](x_y[i])
                        var = self.d_var_encoders[i](x_y[i])
                        d_all_means.append(mean)
                        d_all_vars.append(var)

                d_all_means_mean = torch.stack(d_all_means).mean(0)
                d_all_vars_mean = torch.stack(d_all_vars).mean(0)
                vlb = (d_all_means_mean
                       - x_y.detach()).pow(2).div(d_all_vars_mean) + d_all_vars_mean.log()
                del x_y
                d_reg += vlb.mean() / 2.
                d_reg = d_reg * 0.001

            x_shift = 0.
            if args.use_xs:
                feat_norm = l2normalize(x5)
                y_norm = l2normalize(y5)
                if args.te_idx == -1:
                    feat_norm = feat_norm.view(4, 1536, -1).mean(-1)
                    y_norm = y_norm.view(4, 1536, -1).mean(-1)
                else:
                    feat_norm = feat_norm.view(3, 1536, -1).mean(-1)
                    y_norm = y_norm.view(3, 1536, -1).mean(-1)

                x_shift = get_cond_shift(feat_norm, y_norm.detach(), )
                x_shift = x_shift * 0.0001
                del y_norm


            if args.te_idx == -1:
                n = 4
                # a very brutal way to save memory
                for i in range(n-1):
                    x2, x3, x4, x5 = self.backbone(x)
                    x2, x3, x4, x5 = x2.view(n, -1, 192, 120, 160), x3.view(n, -1, 384, 60, 80), x4.view(n, -1, 768, 30,
                                                                                                         40), \
                        x5.view(n, -1, 1536, 15, 20)
                    gts_shape = gts.shape
                    gts_ = gts.view(n, -1, gts_shape[1], gts_shape[2], gts_shape[3])
                    outs, loss = self._forward_helper(x2[i], x3[i], x4[i], x5[i], gts_[i])
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    del x2, x3, x4, x5, outs, loss, gts_
                    torch.cuda.empty_cache()

                x2, x3, x4, x5 = self.backbone(x)
                x2, x3, x4, x5 = x2.view(n, -1, 192, 120, 160), x3.view(n, -1, 384, 60, 80), x4.view(n, -1, 768, 30,
                                                                                                     40), \
                    x5.view(n, -1, 1536, 15, 20)
                gts_shape = gts.shape
                gts_ = gts.view(n, -1, gts_shape[1], gts_shape[2], gts_shape[3])
                outs, loss = self._forward_helper(x2[-1], x3[-1], x4[-1], x5[-1], gts_[-1])
                return outs, loss + reg_loss + d_reg + x_shift

            else:
                outs, loss = self._forward_helper(x2, x3, x4, x5, gts)
                return outs, loss + reg_loss + d_reg + x_shift
        else:
            x2, x3, x4, x5 = self.backbone(x)
            return self._forward_helper(x2, x3, x4, x5, gts)



    def _forward_helper(self, x2, x3, x4, x5, gts):
        outs = {}
        metric = self.mlayer(x5)
        x = self.up_4(x5, x4)
        d = self.vlayer(x)
        if self.training:
            var_loss = self.var_loss(x, d, gts)
        x, d = self.ref_4(x, d)
        d_u4 = F.interpolate(d, scale_factor=16, mode='bilinear', align_corners=True)
        x = self.up_3(x, x3)
        x, d = self.ref_3(x, F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True))
        d_u3 = F.interpolate(d, scale_factor=8, mode='bilinear', align_corners=True)
        x = self.up_2(x, x2)
        x, d = self.ref_2(x, F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True))
        d_u2 = F.interpolate(d, scale_factor=4, mode='bilinear', align_corners=True)
        d = d_u2 + d_u3 + d_u4
        d = torch.sigmoid(metric[:, 0:1]) * (self.outc(d) + torch.exp(metric[:, 1:2]))
        outs['scale_1'] = d

        if self.training:
            si_loss = self.si_loss(outs, gts)
            return outs['scale_1'], var_loss + si_loss
        else:
            return outs['scale_1']


