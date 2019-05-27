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
import pickledb
import threading
import numpy as np
from openalpr import Alpr

parser = argparse.ArgumentParser(description='Arguments for inside or outside camera.')
parser.add_argument('-IN', '--InGarage', action='store_true', help='Argument to play as camera from inside the parking garage')
parser.add_argument('-OUT', '--OutGarage', action="store_true", help='Argument to play as camera from outside the parking garage')
args = parser.parse_args()
if not args.InGarage and not args.OutGarage:
    print("No args entered, exiting program")
    exit()

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

# checking for pickledb files; if not, creating files and loading them.
def load_dbs(infile, regfile):
    exists_in = os.path.isfile('./' + infile)
    exists_reg = os.path.isfile('./' + regfile)
    if not exists_in:
        print('Creating {0}'.format(infile))
        indb = pickledb.load(infile, False)
    else:
        indb = pickledb.load(infile, False)
    if not exists_reg:
        print('Creating {0}'.format(regfile))
        regdb = pickledb.load(regfile, False)
        regdb.set('ANYNAME', str(time.time()))
    else:
        regdb = pickledb.load(regfile, False)
    return indb, regdb

#creating new still image of found license plate
def rectangle(cords, img):
    x = int(min(cords['coordinates'], key=lambda ev: ev['x'])['x'])
    y = int(min(cords['coordinates'], key=lambda ev: ev['y'])['y'])
    w = int(max(cords['coordinates'], key=lambda ev: ev['x'])['x'])
    h = int(max(cords['coordinates'], key=lambda ev: ev['y'])['y'])
    cv2.rectangle(img,(x,y),(w,h),(255,0,0),2)
    cv2.imshow('Plate Detected', img)

#processing image frame in new thread - TODO look at different methods
def newThread(alpr, num, indb, regdb):
    results = alpr.recognize_ndarray(num)
    if results['results'] != []:
        max_c = max(results['results'], key=lambda ev: ev['confidence'])
        if int(max_c['confidence']) > 80: # adjust confidence here
            print("got hit")
            # if the license plate is in the in-garage database, open the gate and update db's
            if indb.get(max_c['plate']) and args.InGarage:
                indb.rem(max_c['plate'])
                regdb.set(str(max_c['plate']), str(time.time()))
                rectangle(max_c, num)
                print('Opening gate, {0} has left the parking garage'.format(max_c['plate']))
                #print(json.dumps(max_c, indent=4))
            # if the license plate is not in the in-garage database but seen from in the garage
            elif regdb.get(max_c['plate']) and args.InGarage and not indb.get(max_c['plate']):
                if time.time() - float(regdb.get(max_c['plate'])) > 10:
                    print('{0} is a registered vehicle that has already left the garage. Do not open gate.'.format(max_c['plate']))
                    rectangle(max_c, num)
            # if the license plate is registered and seen from the out side camera, and not in the garage database, open gate
            elif regdb.get(max_c['plate']) and args.OutGarage and not indb.get(max_c['plate']):
                print('{0} is a registered vehicle, opening gate'.format(max_c['plate']))
                indb.set(str(max_c['plate']), str(time.time()))
                rectangle(max_c, num)
            # if the license plate is registered and seen from the out side camera but in the in-garage database
            elif regdb.get(max_c['plate']) and args.OutGarage and indb.get(max_c['plate']):
                if time.time() - float(indb.get(max_c['plate'])) > 10:
                    print('{0} is a registered vehicle that is already in the garage, do not open gate'.format(max_c['plate']))
                    rectangle(max_c, num)

def lp_detect() :
    in_db, reg_db = load_dbs('cars_in.db', 'cars_reg.db')
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
                    #if str(frame)[-1] == "0":
                    #    t = threading.Thread(target=newThread, args=(alpr, img, in_db, reg_db))
                    #    t.start()
                    newThread(alpr, img, in_db, reg_db)
                except Exception as e:
                    print('App error: {0}'.format(e))
            cv2.imshow('License Plate Detect',img)
            keyCode = cv2.waitKey(30) & 0xff
            # Stop the program on the ESC key
            if keyCode == 27:
                print('Escape key was pressed')
                in_db.getall()
                in_db.dump()
                reg_db.dump()
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
