import cv2
import mss
import mss.tools
import time
import numpy as np
from pykeyboard import PyKeyboard
from simpleLaneDetect import process_img 
from getKeys import main
import os


k = PyKeyboard()
#import pyautogui

# gives us time to get situated in the game
for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []



with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {'top': 50, 'left': 64, 'width': 512, 'height': 384}

    while 'Screen capturing':
        last_time = time.time()
        cap = 0
        # Get raw pixels from the screen, save it to a Numpy array
        #k.press_key('W')
        img = np.array(sct.grab(monitor))
        screen = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #result = process_img(img)

        # Display the picture
        cv2.imshow('OpenCV/Numpy normal', screen)
        #todo training data
        keys = main()
        print(keys)
        training_data.append([screen,keys])


        print('loop took {} seconds'.format(time.time()-last_time))         
        print('fps: {0}'.format(1 / (time.time()-last_time)))

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name,training_data)       