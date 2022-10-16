import os
from paper.config import BASE_DIR, RAW_DIR, ENH_DIR, RES_DIR

command = f"python train.py \
--input_dir {RAW_DIR}/ \
--experts_dir {ENH_DIR}/ \
--train_list {BASE_DIR}/paper/processing/train_fnames.txt \
--val_list {BASE_DIR}/paper/processing/val_fnames.txt \
--batchsize 10 \
--npoints 10 \
--nfilters 8 \
--expname baseline_rotated \
--logs_dir {RES_DIR}/logs/ \
--models_dir {RES_DIR}/models/ \
"
os.system(command)