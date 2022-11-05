from numpy import index_exp
import pytorch_lightning as pl
from torch import nn
import torch
from abc import ABC, abstractmethod
from torchvision.transforms.functional import resize
from torchvision.utils import make_grid

from backbones import GammaBackbone, SpliNetBackbone


class AdaptiveGammaLUTNet(nn.Module):
    # LUT methods
    def __init__(self):
        super().__init__()
        self.backbone = SpliNetBackbone(n=1, nc=8, n_input_channels=3, n_output_channels=1)

    def get_params(self, x):
        gamma = self.backbone(x)
        return {"gamma": gamma}

    def enhance(self, x, params):
        return x ** params["gamma"]


class SimplestSpline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SpliNetBackbone(n=5, nc=21, n_input_channels=3, n_output_channels=3)

    def get_params(self, x):
        return {"ys": self.backbone(x)}

    def enhance(self, x, params):
        # x is (B, 3, H, W)  params['ys'] is (B, 1, 1, N)
        # something sophisticated
        out = x.clone()
        for channel_ind in range(x.shape[1]):
            out[:, channel_ind] = self.apply_to_one_channel(out[:, channel_ind], params)
        return out
    
    def apply_to_one_channel(self, x, params):
        # x is (B, H, W)
        # params is {'ys': ys} and ys is (B, 1, 1, N=5)
        # something sophisticated
        ys = params['ys'].reshape(params['ys'].shape[0], params['ys'].shape[-1])  # (B, N)
        N = ys.shape[-1]
        xs = torch.linspace(0, 255, N+2)[None]  # (1, N)
        slopes = torch.diff(ys)/(xs[:, 1]-xs[:, 0])
        out = torch.zeros_like(x)
        for i in range(1, N):
            locations = (x < xs[:, i]) * (xs[:, i-1] <= x)
            res = ys[:, i, None, None] - (xs[:, i]-x)*slopes[:, i-1, None, None]
            out[locations] = res[locations]
        return out

class ThinnestPlateSpline(nn.Module):
    def __init__(self, nknots=10):
        super().__init__()
        self.backbone = SpliNetBackbone(n=(2*nknots+1), nc=8, n_input_channels=3, n_output_channels=3)
        self.nknots = nknots

    def get_params(self, x, lambdas_scale=1000):
        nout = self.backbone(x)
        xs = nout[:,:,:,:self.nknots]
        ys = nout[:,:,:,self.nknots:-1]
        ls = nout[:,:,:,-1]
        return {"ys": ys, "xs":xs, "lambdas":ls/lambdas_scale}

    @staticmethod
    def build_k(xs_eval, xs_control):
        # "classic" TPS energy (m=2), null space is just the affine functions span{1, r, g, b} if for instance the dimension of the null space is 3
        # xs_control : (Bx)Nx3
        # xs_eval : (Bx)Mx3
        # returns (Bx)Mx(N+4) matrix
        M = xs_eval.shape[-1]
        d = torch.linalg.norm(
            xs_eval[:,0,:,None] - xs_control[:,0,None], axis=2  # M x 1 x 3  # 1 x N x 3
        )  # M x N x 3
        return torch.hstack((d, torch.ones((M,1)), xs_eval)).requires_grad_()

    @staticmethod
    def build_k_train(xs_control, l):
        # "classic" TPS energy (m=2), null space is just the affine functions span{1, r, g, b} 
        # xs_control : (Bx)Nx3
        # xs_eval : (Bx)Mx3
        # returns (Bx)Mx(N+4) matrix
        M = xs_control.shape[-1]
        dim_null = xs_control.shape[-2]+1
        print("xs_control", xs_control.shape)
        d = torch.linalg.norm(
            xs_control[:,0,:,None] - xs_control[:,0,None], axis=2
        )  #+ l*torch.eye(M)
        print("d", d.shape)
        top = torch.hstack((d, torch.ones((M,1)), xs_control))
        bottom = torch.hstack((torch.vstack((torch.ones((1,M)), xs_control.T)), torch.zeros((dim_null+1,dim_null+1))))
        return torch.vstack((top,bottom)).requires_grad_()

    def predict(self, raw, params):

        '''xs: N x 3 (input color dim) x 3 (output color dim) [control points]
        alphas: (N+4) x 3 (output color dim) [spline weights]
        raw: HxWx3
        '''
        fimg = raw.reshape(-1, raw.shape[2])  # Mx3, flattened image
        out = torch.empty_like(raw.reshape(-1, raw.shape[2]))
        for i in range(raw.shape[2]):
            K_ch_i = self.build_k_train(params['xs'], l=params['lambdas'][:,i])
            K_pred_i = self.build_k(fimg, params['xs'])
            nctrl = len(params['xs'])
            zs = torch.zeros((raw.shape[2]+1,1)).requires_grad_()
            ys = params['ys'][:,i].reshape((nctrl,1))
            out[:,i] = K_pred_i @ torch.linalg.pinv(K_ch_i) @ (torch.cat((ys,zs)).flatten())
        return out.reshape(raw.shape)  # HxWx3

    def enhance(self, x, params, lscale=10000):
        # x is (B, 3, H, W);  params['ys'] is (B, 3, N); params['xs'] is (B, 3, N); params['lambdas'] is (B, 3)
        # we have N total control points -- the same ones in each channel -- and 3 lambdas
        out = x.clone()
        xs = params['xs']
        ys = params['ys']
        ls = params['lambdas']
        print("XS", xs.shape)
        print("YS", ys.shape)
        print("ls", ls.shape)
        out = self.predict(out, params)
        return out



class AverageGammaLUTNet(nn.Module):
    # LUT methods
    def __init__(self):
        super().__init__()
        self.backbone = GammaBackbone()

    def get_params(self, x):
        return {"gamma": self.backbone.gamma}

    def enhance(self, x, params):
        return x ** params["gamma"]


################333333


class LightningLUTNet(pl.LightningModule):
    def __init__(self, lutnet, loss_fn):
        super().__init__()
        self.lutnet = lutnet
        self.loss_fn = loss_fn

    def predict(self, x):
        assert len(x) == 1  # assume x is a tensor of size (1, 3, H, W)
        if x.size(2) == 256 and x.size(3) == 256:
            out = self(x)  # call forward
        else:
            params = self.lutnet.get_params(
                resize(x, (256, 256))
            )  # obtain params with small image
            out = self.lutnet.enhance(x, params)  # enhance large image
        return out

    def forward(self, x):
        params = self.lutnet.get_params(x)
        out = self.lutnet.enhance(x, params)
        return out

    def training_step(self, batch, batch_idx):
        raw, target = batch
        out = self(raw)
        loss = self.loss_fn(out, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        raw, target = batch  # this one has batch size 1
        out = self.predict(
            raw
        )  # we use predict and not forward because of image size
        loss = self.loss_fn(out, target)
        self.log("val_loss", loss)
        input_grid = make_grid(raw)
        out_grid = make_grid(target)
        self.logger.experiment.add_image(f"input_{batch_idx}", input_grid)
        self.logger.experiment.add_image(f"output_{batch_idx}", out_grid)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
