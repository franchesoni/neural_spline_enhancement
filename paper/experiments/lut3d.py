import time
import os
from pathlib import Path

import tqdm
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, device_put, devices
from PIL import Image

from config import ENH_DIR, RAW_DIR, RES_DIR


def find_lut(raw, enh):
  st = time.time()
  f_raw = raw.reshape(-1, 3)
  f_enh = enh.reshape(-1, 3)
  lut = {}
  # for ind, rgb in tqdm.tqdm(enumerate(f_raw)):
  for ind, rgb in enumerate(f_raw):
    trgb = tuple(rgb)
    if trgb in lut:
      lut[trgb].append(f_enh[ind])
    else:
      lut[trgb] = [f_enh[ind]]
  lut = {k: np.mean(v, axis=0) for k, v in lut.items()}
  # print('find_lut time:', time.time() - st)
  return lut

def pdot(ys, affinities):
  return jnp.sum(ys * affinities[..., None], axis=0)

def mnorm(xs, rgb):
  res = xs - rgb
  res = -jnp.linalg.norm(res, axis=1, ord=1)
  return res

jit_pdot = jit(pdot)
jit_mnorm = jit(mnorm)

def trilinear(rgb, xs, ys, T):
  # st = time.time()

  # compute affinities 
  affinities = xs - rgb
  affinities = -jnp.linalg.norm(affinities, axis=1, ord=1)
  # affinities = jit_mnorm(xs, rgb)

  # t1 = time.time() - st
  # print('t1', t1)

  affinities = affinities / T
  affinities = jnp.exp(affinities)

  # t2 = time.time() - st - t1
  # print('t2', t2)

  affinities = affinities / jnp.sum(affinities)

  # t3 = time.time() - st - t1 - t2
  # print('t3', t3)

  # result = jnp.sum(ys * affinities[..., None], axis=0)
  result = jit_pdot(ys, affinities)

  # t4 = time.time() - st - t1 - t2 - t3
  # print('t4', t4)

  return result




jit_vmap_trilinear = jit(vmap(trilinear, in_axes=[0, None, None, None]), device=devices('gpu')[1])
# pmap_vmap_trilinear = pmap(vmap(trilinear, in_axes=[0, None, None, None]),
#                         in_axes=[0, None, None, None])




def apply_lut(raw, lut, T=1, fast=True):
  st = time.time()
  f_raw = raw.reshape(-1, 1, 3)
  xs = jnp.array(list(lut.keys()))  # A x 3
  ys = jnp.array(list(lut.values()))  # A x 3
  f_enhs = []
  if fast:
    B = 8
    for i in range(B):
      f_enh = jit_vmap_trilinear(f_raw[len(f_raw)//B * i:len(f_raw)//B * (i+1)], xs, ys, T)
      f_enhs.append(f_enh)
    if len(f_raw)//B * (i+1) < len(f_raw):
      f_enh = jit_vmap_trilinear(f_raw[len(f_raw)//B * B:len(f_raw)//B * (B+1)], xs, ys, T)
      f_enhs.append(f_enh)
    f_enh = jnp.concatenate(f_enhs, axis=0)
    # f_enh = pmap_vmap_trilinear([f_raw[:len(f_raw)//2], f_raw[len(f_raw)//2:]], xs, ys, T)
    # print('apply_lut time:', time.time() - st)
    return np.array(f_enh.reshape(raw.shape), dtype=np.uint8)
  else:
    f_enh = np.empty_like(f_raw)
    for ind, rgb in tqdm.tqdm(enumerate(f_raw)):
      f_enh[ind] = trilinear(rgb, xs, ys, T)
    # print('apply_lut time:', time.time() - st)
    return f_enh.reshape(raw.shape)

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

    # raws = np.array(Image.open(Path(RAW_DIR) / raw_list[source_ind]))
    # enhs = np.array(Image.open(Path(ENH_DIR) / enh_list[source_ind]))
    # rawd = np.array(Image.open(Path(RAW_DIR) / raw_list[dest_ind]))
    # enhd = np.array(Image.open(Path(ENH_DIR) / enh_list[dest_ind]))

    raw = np.array(Image.open(Path(RAW_DIR) / raw_list[source_ind]))
    enh = np.array(Image.open(Path(ENH_DIR) / enh_list[source_ind]))

    # lut = find_lut(raws, enhs)
    lut = find_lut(raw, enh)
    # e = apply_lut(rawd, lut, T=1, fast=True)
    e = apply_lut(raw, lut, T=1, fast=True)

    # Image.fromarray(raws).save(f'figs/raws.png')
    # Image.fromarray(rawd).save(f'figs/rawd.png')
    # Image.fromarray(enhs).save(f'figs/enhs.png')
    # Image.fromarray(enhd).save(f'figs/enhd.png')
    Image.fromarray(e).save(os.path.join(RES_DIR, 'lut3d', raw_list[source_ind]))

