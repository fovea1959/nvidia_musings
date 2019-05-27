'''
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. For referenced software, check for specific copyright and licenses.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
#sudo systemctl restart nvargus-daemon
# On the Jetson Nano, OpenCV comes preinstalled
# Data files are in /usr/sharc/OpenCV
import os
import sys
import cv2
import json
import time
import argparse
import numpy as np
import threading
from openalpr import Alpr

# def gstreamer_pipeline credit is from https://github.com/JetsonHacksNano/CSI-Camera/blob/master/simple_camera.py
# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 30fps 
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen
def gstreamer_pipeline (capture_width=3280, capture_height=2464, display_width=820, display_height=616, framerate=30, flip_method=0) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

#creating new still image of found license plate
def rectangle(cords, img):
    x = int(min(cords['coordinates'], key=lambda ev: ev['x'])['x'])
    y = int(min(cords['coordinates'], key=lambda ev: ev['y'])['y'])
    w = int(max(cords['coordinates'], key=lambda ev: ev['x'])['x'])
    h = int(max(cords['coordinates'], key=lambda ev: ev['y'])['y'])
    cv2.rectangle(img,(x,y),(w,h),(255,0,0),2)
    cv2.imshow('Plate Detected', img)

def look_at_plate(alpr, num):
    print("got frame")
    results = alpr.recognize_ndarray(num)
    if results['results'] != []:
        max_c = max(results['results'], key=lambda ev: ev['confidence'])
        if int(max_c['confidence']) > 80: # adjust confidence here
            rectangle(max_c, num)
            print('Plate {0} detected'.format(max_c['plate']))
            print(json.dumps(max_c, indent=4))

def lp_detect() :
    parser = argparse.ArgumentParser(description='Arguments for inside or outside camera.')
    parser.add_argument('--threaded', action='store_true', help='Put each frame on it\'s own thread')
    args = parser.parse_args()

    alpr = Alpr("us", "/etc/openalpr/openalpr.conf", "/usr/share/openalpr/runtime_data")
    if not alpr.is_loaded():
        print("Error loading OpenALPR")
        sys.exit(1)
    alpr.set_top_n(2)
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        frame = 0
        cv2.namedWindow('License Plate Detect', cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty('License Plate Detect',0) >= 0:
            ret, img = cap.read()
            if ret:
                frame+=1
                try:
                    if args.threaded:
                        t = threading.Thread (target=look_at_plate, args=(alpr, img))
                        t.start()
                    else:
                        look_at_plate(alpr, img)
                except Exception as e:
                    print('App error: {0}'.format(e))
                cv2.imshow('License Plate Detect',img)
            keyCode = cv2.waitKey(30) & 0xff
            # Stop the program on the ESC key
            if keyCode == 27:
                print('Escape key was pressed')
                time.sleep(1)
                break
        print('Closing cap')
        cap.release()
        print('Closing windows')
        cv2.destroyAllWindows()
        time.sleep(1)
        print('Closing alpr')
        alpr.unload()
    else:
        print("Unable to open camera")
if __name__ == '__main__':
    lp_detect()
