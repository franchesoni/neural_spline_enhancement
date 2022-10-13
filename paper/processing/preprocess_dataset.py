import os
import glob

from PIL import Image
import tqdm

from config import ORAW_DIR, OENH_DIR

oraw_list = glob.glob(os.path.join(ORAW_DIR, '*.jpg'))
oenh_list = glob.glob(os.path.join(OENH_DIR, '*.jpg'))

for img_path in tqdm.tqdm(oraw_list + oenh_list):
  img = Image.open(img_path)
  if img.size[0] != 500:
    img = img.transpose(Image.Transpose.ROTATE_90)
  if img.size != (500, 333):
    img = img.resize((500, 333))
  img.save(img_path.replace('_original', ''))

