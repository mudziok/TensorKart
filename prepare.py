import numpy as np
from skimage.transform import resize
from skimage.io import imread
import glob
import os

def resize_image(img):
    im = resize(img, (120, 160, 3))
    im_arr = im.reshape((120, 160, 3))
    return im_arr

print('Loading folders')

paths = glob.glob("data\*\inputs.csv")

for path in paths:
    base = os.path.dirname(path)
    print('Preparing sample: ' + base)
    
    X = []
    Y = []

    image_files = np.loadtxt(path, delimiter=';', dtype=str, usecols=(0,))
    joystick_values = np.loadtxt(path, delimiter=';', usecols=(1,2))

    Y.append(joystick_values)
    Y = np.concatenate(Y)

    for image_file in image_files:
        image = imread(image_file)
        vec = resize_image(image)
        X.append(vec)

    X = np.asarray(X)
    
    np.save(base + "/X", X)
    np.save(base + "/Y", Y)

print('Done!')
