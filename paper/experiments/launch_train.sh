RAW_DIR="/home/franchesoni/adisk/datasets/mit5k/C/train/raw"; \
ENH_DIR="/home/franchesoni/adisk/datasets/mit5k/C/train/target"; \
SPLINES_DIR="/home/franchesoni/projects/current/splines"; \
python train.py \
--input_dir $RAW_DIR/ \
--experts_dir $ENH_DIR/ \
--train_list $SPLINES_DIR/train_fnames.txt \
--val_list $SPLINES_DIR/val_fnames.txt \
--batchsize 10 \
--npoints 10 \
--nfilters 8 \
# --nexperts 1