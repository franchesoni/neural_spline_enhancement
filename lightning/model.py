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
        self.backbone = SpliNetBackbone(n=5, nc=8, n_input_channels=3)

    def get_params(self, x):
        return {"ys": self.backbone(x)}

  def enhance(self, x, params):
    # x is (B, 3, H, W)
    # something sophisticated
	out = x.clone()
    for channel_ind in range(x.shape[1]):
      out[:, channel_ind] = self.apply_to_one_channel(out[:, channel_ind], params)
	return out
  
  def apply_to_one_channel(self, x, params):
    # x is (B, H, W)
    # params is {'ys': ys} and ys is (B, N=5)
    # something sophisticated
    y_ctrl_vals = params['ys']
    x_ctrl_vals = np.linspace(0, 255, len(y_ctrl_vals)+2)
    slopes = np.linalg.diff(y_ctrl_vals)/(x_ctrl_vals[1]-x_ctrl_vals[0])
    out = y_ctrl_vals[1] - nn.functional.relu(x_ctrl_vals[1]-x)*slopes[0]
    for i in range(2,len(y_ctrl_vals)):
        out += y_ctrl_vals[i] - nn.functional.relu(x_ctrl_vals[i]-x)*slopes[i-1]
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
