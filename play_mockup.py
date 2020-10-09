import numpy as np
import cv2

from train import create_model

x_train = np.load("data/X.npy")
y_train = np.load("data/y.npy")

white = (255, 255, 255)
black = (0, 0, 0)

width = 640
height = 480

radius = 80

model = create_model(keep_prob=1)
model.load_weights('model_weights.h5')

frame = 0
for data in x_train:
    big = cv2.resize(data, (int(width), int(height)), interpolation = cv2.INTER_AREA)
    
    vec = np.expand_dims(data, axis=0)
    prediction = model.predict(vec, batch_size=1)[0]
    circle_x = prediction[0]
    circle_y = prediction[1]

    coords = (int(640/4 + (circle_x * radius)), int(480/2 - (circle_y * radius)))
    final = cv2.line(big, (int(width/4), int(height/2)), coords, black, 5) 
    final = cv2.circle(final, coords, 20, white, -1)

    circle_x = y_train[frame][0]
    circle_y = y_train[frame][1]
    coords = (int(640/4*3 + (circle_x * radius)), int(480/2 - (circle_y * radius)))
    final = cv2.line(big, (int(width/4*3), int(height/2)), coords, black, 5) 
    final = cv2.circle(final, coords, 20, white, -1)


    
    cv2.imshow('test', final)
    
    frame += 1
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
