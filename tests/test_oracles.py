from PIL import Image
import numpy as np

from oracles import LUT3DOracle

def test_LUT3DOracle(raw, enh):
  oracle = LUT3DOracle()
  params = oracle.fit(raw, enh)
  out = oracle.predict(raw, params)
  Image.fromarray(out).save('tests/oracle_LUT3D.png')
  print('RGB MSE:', np.linalg.norm(enh - out))


if __name__ == '__main__':
  raw_path = 'tests/raw_000014.jpg'
  enh_path = 'tests/enh_000014.jpg'
  raw, enh = np.array(Image.open(raw_path)), np.array(Image.open(enh_path))
  test_LUT3DOracle(raw, enh)
