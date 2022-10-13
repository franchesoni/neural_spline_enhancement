test_list_path = "/home/franchesoni/adisk/datasets/mit5k/C/random250.txt"

SEED = 0
N_TRAINVAL = 100
val_ratio = 0.05

import random

N_VAL = int(N_TRAINVAL * val_ratio)
N_TRAIN = N_TRAINVAL - N_VAL

all_filenames = list(map(str, range(5000)))
with open(test_list_path, 'r') as f:
  # remove \n and first line
  test_filenames = f.readlines()[1:]
test_filenames = [f.split('\n')[0] for f in test_filenames]

trainval_filenames = [f for f in all_filenames if f not in test_filenames]  # not efficient but useful
random.Random(SEED).shuffle(trainval_filenames)  
# breakpoint()
format = lambda x: str(x).rjust(6, '0') + '.jpg\n'
train_filenames = list(map(format, trainval_filenames[:N_TRAIN]))
val_filenames = list(map(format, trainval_filenames[N_TRAIN:N_TRAINVAL]))
test_filenames = list(map(format, test_filenames))

with open('train_fnames.txt', 'w') as f:
  f.writelines(train_filenames)
with open('val_fnames.txt', 'w') as f:
  f.writelines(val_filenames)
with open('test_fnames.txt', 'w') as f:
  f.writelines(test_filenames)

