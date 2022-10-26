import pytorch_lightning as pl
from torch import nn
import torch
from abc import ABC, abstractmethod
from torchvision.transforms.functional import resize
from torchvision.utils import make_grid

from backbones import GammaBackbone


class AverageGammaLUTNet(nn.Module):
  # LUT methods
  def __init__(self):
    super().__init__()
    self.backbone = GammaBackbone()

  def get_params(self, x):
    return {'gamma': self.backbone.gamma}

  def enhance(self, x, params):
    return x ** params['gamma']


################333333

class LightningLUTNet(pl.LightningModule):
  def __init__(self, lutnet, loss_fn):
    super().__init__()
    self.lutnet = lutnet 
    self.loss_fn =loss_fn

  def predict(self, x):
    assert len(x) == 1  # assume x is a tensor of size (1, 3, H, W)
    if x.size(2) == 224 and x.size(3) == 224:
      out = self(x)  # call forward
    else:
      params = self.lutnet.get_params(resize(x, (224, 224)))  # obtain params with small image
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
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    raw, target = batch  # this one has batch size 1
    out = self.predict(raw)  # we use predict and not forward because of image size
    loss = self.loss_fn(out, target)
    self.log('val_loss', loss)
    input_grid = make_grid(raw)
    out_grid = make_grid(target)
    self.logger.experiment.add_image(f"input_{batch_idx}", input_grid)
    self.logger.experiment.add_image(f"output_{batch_idx}", out_grid)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    return optimizer


