import numpy.random

import torch
import torch.nn as nn
import torch.nn.functional as F

numpy.random.seed(0)
torch.manual_seed(0)
import matplotlib.pyplot as plt
import numpy as np


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
    C = torch.matmul(a.T, b) / N
    if invert:
        return torch.linalg.pinv(C)
    else:
        return C


def get_cond_shift(X1, Y1, estimator=sample_covariance):
    m1 = torch.mean(X1, dim=0)
    my1 = torch.mean(Y1, dim=0)
    x1 = X1 - m1
    y1 = Y1 - my1

    c_x1_y = estimator(x1, y1)
    c_y_x1 = estimator(y1, x1)

    inv_c_y_y = estimator(y1, y1, invert=True)
    shift = torch.matmul(c_x1_y, torch.matmul(inv_c_y_y, c_y_x1))
    return nn.MSELoss()(shift, torch.zeros_like(shift))


class MyModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.lx_1 = nn.Linear(2, 5)
        self.lx_2 = nn.Linear(5, 5)
        self.lx_3 = nn.Linear(5, 2)

        self.d_mean_encoders = nn.ModuleList([
            MeanEncoder(shape=[1, 2 * 2]) for _ in range(3)
        ])

        self.d_var_encoders = nn.ModuleList([
            VarianceEncoder(shape=[1, 2 * 2]) for _ in range(3)
        ])

        self.ly_1 = nn.Linear(2, 5)
        self.ly_2 = nn.Linear(5, 5)
        self.ly_3 = nn.Linear(5, 2)

        self.C = nn.Sequential(nn.Linear(2, 2))

        self.d_shift = 0.1
        self.y_map = False
        self.shift = 0.

    def forward(self, all_x, all_y, warmup=True):
        x_loss = 0.
        y_loss = 0.
        x_ = []
        y_ = []
        d_all_means = []
        d_all_vars = []

        res = {}
        for i in range(len(all_x)):

            x = self._x_forward(all_x[i])
            pred = self.C(x)
            x_.append(x)
            x_loss += F.mse_loss(pred, all_y[i])

            if self.d_shift > 0 and not warmup:
                y = self._y_forward(all_y[i])
                y_.append(y)
                pred_y = self.C(y)
                y_loss += F.mse_loss(pred_y, all_y[i])
                res['y_loss'] = y_loss

                d_mean = self.d_mean_encoders[i](torch.cat([x, y], dim=-1))
                d_var = self.d_var_encoders[i](torch.cat([x, y], dim=-1))
                d_all_means.append(d_mean)
                d_all_vars.append(d_var)

        res['x_loss'] = x_loss
        d_reg = 0.
        if (self.d_shift > 0 or self.shift > 0) and not warmup:
            x_, y_ = torch.stack(x_), torch.stack(y_)

        if self.d_shift > 0 and not warmup:
            d_all_means_mean = torch.stack(d_all_means).reshape(-1, 4).mean(0)
            d_all_vars_mean = torch.stack(d_all_vars).reshape(-1, 4).mean(0)
            feat_y = torch.cat([x_, y_.detach()], dim=-1)
            vlb = (d_all_means_mean
                   - feat_y).pow(2) + d_all_vars_mean.log()

            d_reg += vlb.mean() / 2.
            d_reg = d_reg * self.d_shift
            res['d_reg'] = d_reg

        return res

    def pred(self, x):
        x = self._x_forward(x)
        pred = self.C(x)
        return pred

    def _x_forward(self, x):
        x = self.lx_1(x)
        x = self.lx_2(x)
        x = self.lx_3(x)
        # x = self.lx_4(x)
        return x

    def _y_forward(self, y):
        if self.y_map:
            y = self.ly_1(y)
            y = self.ly_2(y)
            y = self.ly_3(y)
        else:
            y = y
        return y


def hidden_gt(x):
    x = x
    return x

def construct_domain1(x, y):
    x2 = x + torch.randn(y.shape) * 0.3
    y2 = y + torch.randn(y.shape) * 0.3
    x, y = torch.stack([x, x2], dim=1), torch.stack([y, y2], dim=1)
    return x, y

def construct_domain2(x, y, ):
    x2 = 4 * x**3 + 0.5 + torch.randn(y.shape) * 0.3
    y2 = 4 * x**2 + 0.3
    x, y = torch.stack([x, x2], dim=1), torch.stack([y, y2], dim=1)
    return x,y

def construct_domain3(x, y, ):
    x2 = 2 * x**2 - 0.3 + torch.randn(y.shape) * 0.2
    y2 = 0.5 * x**3 - 0.2
    x, y = torch.stack([x, x2], dim=1), torch.stack([y, y2], dim=1)
    return x,y

def demo():
    hidden_x = torch.randn(10000)
    hidden_y = hidden_gt(hidden_x)
    plt.scatter(hidden_x, hidden_y, c='pink', alpha=1)
    plt.show()

    x1, y1 = construct_domain1(hidden_x, hidden_y)
    x2, y2 = construct_domain2(hidden_x, hidden_y)
    x3, y3 = construct_domain3(hidden_x, hidden_y)
    plt.scatter(x1.numpy().T[0], x1.numpy().T[1],  alpha=1, color='r', marker ='+', label='Unseen X')
    plt.scatter(x2.numpy().T[0], x2.numpy().T[1],  alpha=1, color='b', marker ='+', label='Seen X1')
    plt.scatter(x3.numpy().T[0], x3.numpy().T[1],  alpha=1, color='m', marker ='+', label='Seen X1')
    plt.legend()
    plt.title('Synthetic data: X in raw space.')
    plt.savefig('res_img/Synthetic_X.jpg')
    plt.show()

    plt.scatter(y1.numpy().T[0], y1.numpy().T[1],  alpha=1, color='r', marker ='.', label='Unseen Y')
    plt.scatter(y2.numpy().T[0], y2.numpy().T[1],  alpha=1, color='b', marker ='.', label='Seen Y1')
    plt.scatter(y3.numpy().T[0], y3.numpy().T[1],  alpha=1, color='m', marker ='.', label='Seen Y2')
    plt.legend()
    plt.title('Synthetic data: Y in raw space.')
    plt.savefig('res_img/Synthetic_Y.jpg')
    plt.show()

    hidden_x_v = torch.randn(100)
    hidden_y_v = hidden_gt(hidden_x_v)
    x_v2, y_v2 = construct_domain2(hidden_x_v, hidden_y_v)

    hidden_x_v = torch.randn(100)
    hidden_y_v = hidden_gt(hidden_x_v)
    x_v3, y_v3 = construct_domain3(hidden_x_v, hidden_y_v)

    model = MyModel()
    optim = torch.optim.Adam(model.parameters(), lr=5e-1)
    best_eval = 10**9
    best_test = 10**9
    best_test2 = 10 ** 9

    x_s = torch.stack([x2, x3])
    y_s = torch.stack([y2, y3])
    x_v =torch.stack([x_v2, x_v3])
    y_v =torch.stack([y_v2, y_v3])
    for i in range(500+1):
        model.train()
        all_loss = 0.
        print_string = ''
        res = model(x_s, y_s, warmup = False)
        for k in res:
            all_loss += res[k]
            print_string += f'{k},{res[k].item():.4f}; '
        optim.zero_grad()
        all_loss.backward()
        optim.step()

        print_string += f'all_loss,{all_loss.item():.4f}; '

        model.eval()
        X_v_pred = model.pred(x_v)
        val_loss = F.mse_loss(X_v_pred, y_v)


        best_eval = val_loss
        pred = model.pred(x1)
        test_loss = F.mse_loss(pred, y1, reduction='sum')
        test_loss2 = F.mse_loss(pred, y1, reduction='mean')
        print(f'epoch {i} current test_loss: {test_loss.item():.4f} {test_loss2.item():.4f}')

        if test_loss < best_test:
            x1_h = model._x_forward(x1)
            x2_h = model._x_forward(x2)
            x3_h = model._x_forward(x3)

            plt.scatter(x1_h.detach().numpy().T[0], x1_h.detach().numpy().T[1], alpha=1, color='r', marker='+')
            plt.scatter(x2_h.detach().numpy().T[0], x2_h.detach().numpy().T[1], alpha=1, color='b', marker='+')
            plt.scatter(x3_h.detach().numpy().T[0], x3_h.detach().numpy().T[1], alpha=1, color='m', marker='+')
            title = f'Validation Loss: {val_loss:.4f}; Test Loss: {test_loss2:.4f}'
            plt.title(title)
            plt.savefig(f'res_img/{i}_X.jpg')
            plt.show()

            y1_h = model._y_forward(y1)
            y2_h = model._y_forward(y2)
            y3_h = model._y_forward(y3)

            plt.scatter(y1_h.detach().numpy().T[0], y1_h.detach().numpy().T[1], marker='.', alpha=1, color='r')
            plt.scatter(y2_h.detach().numpy().T[0], y2_h.detach().numpy().T[1], marker='.', alpha=1, color='b')
            plt.scatter(y3_h.detach().numpy().T[0], y3_h.detach().numpy().T[1], marker='.', alpha=1, color='m')
            title = f'Validation Loss: {val_loss:.4f}; Test Loss: {test_loss2:.4f}'
            plt.title(title)
            plt.savefig(f'res_img/{i}_Y.jpg')
            plt.show()
            print('saved image')

            best_test = test_loss
            best_test2 = test_loss2

    print(f'final_val_loss: {best_eval.item():.4f}')
    print(f'final_test_loss: {best_test.item():.4f}')
    print(f'final_test_loss: {best_test2.item():.4f}')
    pred = model.pred(x1)
    test_loss = F.mse_loss(pred, y1, reduction='sum')
    test_loss2 = F.mse_loss(pred, y1, reduction='mean')
    print(f'epoch FINAL current test_loss: {test_loss.item():.4f} {test_loss2.item():.4f}')

if __name__ == '__main__':
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    demo()
