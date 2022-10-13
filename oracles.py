from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap

from paper.config import DEVICE
# given two images, find the parameters of the optimal transformation between them

class AbstractOracle(ABC):
  @abstractmethod
  def fit(self, raw, enh):
    raise NotImplementedError

  @abstractmethod
  def predict(self, raw):
    raise NotImplementedError


####### 3d lut #########
def pdot(ys, affinities):
  return jnp.sum(ys * affinities[..., None], axis=0)

jit_pdot = jit(pdot)

def trilinear(rgb, xs, ys, T):
  affinities = xs - rgb
  affinities = -jnp.linalg.norm(affinities, axis=1, ord=1)
  affinities = affinities / T
  affinities = jnp.exp(affinities)
  affinities = affinities / jnp.sum(affinities)
  result = jit_pdot(ys, affinities)
  return result

jit_vmap_trilinear = jit(vmap(trilinear, in_axes=[0, None, None, None]), device=DEVICE)


class LUT3DOracle(AbstractOracle):
  def __init__(self, T=1, B=8, fast=True):
    self.T = T
    self.B = B
    self.fast = fast

  def fit(self, raw, enh):
    f_raw = raw.reshape(-1, 3)
    f_enh = enh.reshape(-1, 3)
    lut = {}
    for ind, rgb in enumerate(f_raw):
      trgb = tuple(rgb)
      if trgb in lut:
        lut[trgb].append(f_enh[ind])
      else:
        lut[trgb] = [f_enh[ind]]
    lut = {k: np.mean(v, axis=0) for k, v in lut.items()}
    self.params = lut
    return lut

  def predict(self, raw, params=None):
    params = params or self.params
    f_raw = raw.reshape(-1, 1, 3)
    xs = jnp.array(list(self.params.keys()))  # A x 3
    ys = jnp.array(list(self.params.values()))  # A x 3
    f_enhs = []
    if self.fast:
      for i in range(self.B):
        f_enh = jit_vmap_trilinear(f_raw[len(f_raw)//self.B * i:len(f_raw)//self.B * (i+1)], xs, ys, self.T)
        f_enhs.append(f_enh)
      if len(f_raw)//self.B * (i+1) < len(f_raw):
        f_enh = jit_vmap_trilinear(f_raw[len(f_raw)//self.B * self.B:len(f_raw)//self.B * (self.B+1)], xs, ys, self.T)
        f_enhs.append(f_enh)
      f_enh = jnp.concatenate(f_enhs, axis=0)
      return np.array(f_enh.reshape(raw.shape), dtype=np.uint8)
    else:
      f_enh = np.empty_like(f_raw)
      for ind, rgb in enumerate(f_raw):
        f_enh[ind] = trilinear(rgb, xs, ys, self.T)
      return f_enh.reshape(raw.shape)


