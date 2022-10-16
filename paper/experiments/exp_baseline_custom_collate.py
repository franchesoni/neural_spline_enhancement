import os
from paper.config import BASE_DIR, ORAW_DIR, OENH_DIR, RES_DIR

command = f"python train.py \
--input_dir {ORAW_DIR}/ \
--experts_dir {OENH_DIR}/ \
--train_list {BASE_DIR}/paper/processing/train_fnames.txt \
--val_list {BASE_DIR}/paper/processing/val_fnames.txt \
--batchsize 10 \
--npoints 10 \
--nfilters 8 \
--logs_dir {RES_DIR}/logs/ \
--models_dir {RES_DIR}/models/ \
--custom_collate \
--expname baseline_custom_collate \
"
os.system(command)