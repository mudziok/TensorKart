import numpy as np
import cv2
from mss import mss
from PIL import Image
from skimage.transform import resize
from skimage.io import imread
import math
import pyvjoy
from threading import Thread
from inputs import get_gamepad

from train import create_model

def resize_image(img):
    im = resize(img, (120, 160, 3))
    im_arr = im.reshape((120, 160, 3))
    return im_arr

coords = {'top': 34, 'left':3, 'width':640, 'height':480}
sct = mss()

white = (255, 255, 255)
black = (0, 0, 0)

width = 640
height = 480

radius = 80

j = pyvjoy.VJoyDevice(2)
MAX_VJOY = 32767

MAX_JOY_VAL = math.pow(2, 15)
joy_x = 0
joy_y = 0
manual = False

def gamepadThread():
    global joy_x
    global joy_y
    global manual
    while True:
        events = get_gamepad()
    
        for event in events:
            if (event.code == 'ABS_Y'):
                joy_y = event.state / MAX_JOY_VAL
            if (event.code == 'ABS_X'):
                joy_x = event.state / MAX_JOY_VAL
            if (event.code == 'BTN_TL' and event.state == True):
                manual = not manual
                state = ("off", "on")[manual]
                print('Toggled manual ' + state)

captureGamepad = Thread(target = gamepadThread)
captureGamepad.start()

model = create_model(keep_prob=1)
model.load_weights('model_weights.h5')

frame = 0
while True:
    sct_img = sct.grab(coords)
    img_bgr = Image.frombytes('RGB', (sct_img.size.width, sct_img.size.height), sct_img.rgb)
    small = cv2.resize(np.array(img_bgr), (160, 120), interpolation = cv2.INTER_AREA)

    resized = resize_image(small)
    
    vec = np.expand_dims(resized, axis=0)
    prediction = model.predict(vec, batch_size=1)[0]
    circle_x = prediction[0]
    circle_y = prediction[1]

    j.data.wAxisX = int(((circle_x / 2) + 0.5) * MAX_VJOY)
    j.data.wAxisY = int(((-circle_y / 2) + 0.5) * MAX_VJOY)
    j.update()

    big = cv2.resize(resized, (int(width), int(height)), interpolation = cv2.INTER_AREA)

    ball = (int(width/2 + (circle_x * radius)), int(height/2 - (circle_y * radius)))
    final = cv2.line(big, (int(width/2), int(height/2)), ball, black, 5) 
    final = cv2.circle(final, ball, 20, white, -1)
    
    cv2.imshow('test', big)
    
    frame += 1
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
