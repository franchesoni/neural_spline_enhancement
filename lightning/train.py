from torch.nn import MSELoss
import pytorch_lightning as pl


from model import LightningLUTNet, AverageGammaLUTNet
from data import FiveKDataModule

def train_average_gamma():
  lutnet = AverageGammaLUTNet()
  PLlutnet = LightningLUTNet(lutnet, loss_fn=MSELoss())
  trainer = pl.Trainer(fast_dev_run=False, overfit_batches=1, max_time='0:0:0:30', log_every_n_steps=1)
  trainer.fit(PLlutnet, datamodule=FiveKDataModule(batch_size=8, transform='resize'))

if __name__ == '__main__':
  train_average_gamma()
  

