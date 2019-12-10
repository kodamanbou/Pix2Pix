from PIL import Image
import os
import glob
from tqdm import tqdm

for filename in tqdm(glob.glob('./datasets/*.jpg')):
    try:
        image = Image.open(filename)
    except:
        print('Error !!!')
        os.remove(filename)
