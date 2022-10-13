from PIL import Image
import numpy as np

from oracles import LUT3DOracle, PerChannelOracle

def test_LUT3DOracle(raw, enh):
  oracle = LUT3DOracle()
  params = oracle.fit(raw, enh)
  out = oracle.predict(raw, params)
  Image.fromarray(out).save('tests/oracle_LUT3D.png')
  print('RGB MSE:', np.linalg.norm(enh - out))

def test_PerChannelOracle(raw, enh):
  oracle = PerChannelOracle()
  params = oracle.fit(raw, enh)
  out = oracle.predict(raw, params).astype(np.uint8)
  Image.fromarray(out).save('tests/oracle_PerChannel.png')
  print('RGB MSE:', np.linalg.norm(enh - out))


if __name__ == '__main__':
  S = 1000
  raw_path = 'tests/raw_000014.jpg'
  enh_path = 'tests/enh_000014.jpg'
  raw, enh = np.array(Image.open(raw_path))[:S, :S], np.array(Image.open(enh_path))[:S, :S]

  test_LUT3DOracle(raw, enh)
  test_PerChannelOracle(raw, enh)

  # from ptcolor import rgb2lab, lab2rgb
  # import torch
  # raw_lab = rgb2lab(torch.Tensor(raw).permute(2, 0, 1)[None])
  # raw2 = lab2rgb(raw_lab)[0].permute(1, 2, 0)
  # breakpoint()
