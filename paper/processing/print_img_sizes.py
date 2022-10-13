import os

import numpy as np
from PIL import Image

from config import RAW_DIR, ENH_DIR


with open('train_fnames.txt', 'r') as f:
  train_fnames = f.readlines()
  train_fnames = [n.split('\n')[0] for n in train_fnames]

aa = []
for img_name in train_fnames:
  aa.append(min(np.array(Image.open(os.path.join(RAW_DIR, img_name))).shape[:2]))
  aa.append(min(np.array(Image.open(os.path.join(ENH_DIR, img_name))).shape[:2]))

with open('val_fnames.txt', 'r') as f:
  train_fnames = f.readlines()
  train_fnames = [n.split('\n')[0] for n in train_fnames]

bb = []
for img_name in train_fnames:
  bb.append(min(np.array(Image.open(os.path.join(RAW_DIR, img_name))).shape[:2]))
  bb.append(min(np.array(Image.open(os.path.join(ENH_DIR, img_name))).shape[:2]))




print(min(aa))
print(max(aa))
print(np.unique(aa, return_counts=True))
print(min(bb))
print(max(bb))
print(np.unique(bb, return_counts=True))