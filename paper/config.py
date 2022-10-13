from pathlib import Path
ORAW_DIR = '/home/franchesoni/adisk/datasets/mit5k/C/train_original/raw'
OENH_DIR = '/home/franchesoni/adisk/datasets/mit5k/C/train_original/target'

RAW_DIR = '/home/franchesoni/adisk/datasets/mit5k/C/train/raw'
ENH_DIR = '/home/franchesoni/adisk/datasets/mit5k/C/train/target'

RES_DIR = "/home/franchesoni/adisk/results/splines/"

Path(RAW_DIR).mkdir(exist_ok=True, parents=True)
Path(ENH_DIR).mkdir(exist_ok=True, parents=True)

RAW_DIR = '/home/franchesoni/data/mit5k/dnr/C/train/raw/'
ENH_DIR = '/home/franchesoni/data/mit5k/dnr/C/train/target/'

from jax import devices
DEVICE = devices('cpu')[0]
# DEVICE = devices('gpu')[0]
