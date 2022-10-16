import time
import os
from pathlib import Path

import tqdm
import numpy as np
from PIL import Image
import jax.numpy as jnp
from jax import jit, vmap, devices

from config import ENH_DIR, RAW_DIR, RES_DIR

DEVICE = devices('gpu')[1]

def find_channel_lut(f_raw, f_enh):
  lut = {}
  for ind, cvalue in enumerate(f_raw):
    if cvalue in lut:
      lut[cvalue].append(f_enh[ind])
    else:
      lut[cvalue] = [f_enh[ind]]
  lut = {k: np.mean(v, axis=0) for k, v in lut.items()}
  return lut

def find_per_channel_lut(raw, enh):
  f_raw = raw.reshape(-1, 3)
  f_enh = enh.reshape(-1, 3)
  lut = {}
  for ci, c in enumerate(['r', 'g', 'b']):
    lut[c] = find_channel_lut(f_raw[:, ci], f_enh[:, ci])
  return lut







def pdot(ys, affinities):
  return jnp.sum(ys * affinities, axis=0)

jit_pdot = jit(pdot)

def interpolate_single(value, xs, ys, T):
  affinities = - jnp.abs(xs.astype(float) - value)
  affinities = affinities / T
  affinities = jnp.exp(affinities)
  affinities = affinities / jnp.sum(affinities)
  result = jit_pdot(ys, affinities)
  return result

jit_vmap_interpolate = jit(vmap(interpolate_single, in_axes=[0, None, None, None]), device=DEVICE)
# jit_vmap_interpolate = vmap(interpolate_single, in_axes=[0, None, None, None])



def apply_per_channel_lut(raw, lut, T=1):
  f_raw = raw.reshape(-1, 3)
  f_enh = []
  for ci, c in enumerate(['r', 'g', 'b']):
    f_raw_c = f_raw[:, ci]
    lut_c = lut[c]
    xs, ys = jnp.array(list(lut_c.keys())), jnp.array(list(lut_c.values()))
    f_enh_c = jit_vmap_interpolate(f_raw_c, xs, ys, T)
    f_enh.append(f_enh_c)
  f_enh = np.stack(f_enh, axis=1)
  return f_enh

  

def ptensor(x):
  print(type(x), x.dtype)
  print(x.shape, x.min(), x.max())

 
if __name__ == '__main__':
  raw_list = sorted(os.listdir(RAW_DIR))
  enh_list = sorted(os.listdir(ENH_DIR))
  assert raw_list == enh_list

  for ind in tqdm.tqdm(range(len(raw_list))):
    source_ind = ind
    dest_ind = ind

    raw = np.array(Image.open(Path(RAW_DIR) / raw_list[source_ind]))
    enh = np.array(Image.open(Path(ENH_DIR) / enh_list[source_ind]))

    lut = find_per_channel_lut(raw, enh)
    e = apply_per_channel_lut(raw, lut).reshape(raw.shape)
    hist = apply_per_channel_lut(np.stack([np.arange(256)]*3, axis=1), lut, T=10)
    import matplotlib.pyplot as plt
    for ci, c in enumerate(['r', 'g', 'b']):
      plt.figure()
      plt.plot(raw.reshape(-1, 3)[:, ci], enh.reshape(-1, 3)[:, ci], '.')
      plt.plot(np.arange(256), hist[:, ci], 'o')
      plt.plot(np.arange(256), np.arange(256), '.')
      plt.plot(lut[c].keys(), lut[c].values(), '.')
      plt.savefig(f'{c}.png')

    # Image.fromarray(raws).save(f'figs/raws.png')
    # Image.fromarray(rawd).save(f'figs/rawd.png')
    # Image.fromarray(enhs).save(f'figs/enhs.png')
    # Image.fromarray(enhd).save(f'figs/enhd.png')
    dest_dir = os.path.join(RES_DIR, 'per_channel')
    Path(dest_dir).mkdir(exist_ok=True, parents=True)
    Image.fromarray(e.astype(np.uint8)).save(os.path.join(dest_dir, raw_list[source_ind]))

