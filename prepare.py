import numpy as np
from skimage.transform import resize
from skimage.io import imread

def resize_image(img):
    im = resize(img, (120, 160, 3))
    im_arr = im.reshape((120, 160, 3))
    return im_arr

print('Preparing data')

X = []
Y = []

image_files = np.loadtxt('data/inputs.csv', delimiter=';', dtype=str, usecols=(0,))
joystick_values = np.loadtxt('data/inputs.csv', delimiter=';', usecols=(1,2))

Y.append(joystick_values)
Y = np.concatenate(Y)

for image_file in image_files:
    image = imread(image_file)
    vec = resize_image(image)
    X.append(vec)

X = np.asarray(X)

print('Saving data')

np.save("data/X", X)
np.save("data/Y", Y)

print('Done!')

