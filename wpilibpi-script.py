
# Import the camera server
import json
import math
import numbers
import os
from threading import Thread
from time import sleep
from networktables import NetworkTables
import cscore

# Import OpenCV and NumPy
import cv2
import numpy as np

DIM=(1280, 720)
K=np.array([[542.7249909542755, 0.0, 649.0068172136788], [0.0, 541.5532083705964, 420.73848083674886], [0.0, 0.0, 1.0]])
D=np.array([[-0.07366601228007531], [0.04484958515662308], [-0.030682029931240723], [0.006384847748248899]])

def main():
    # *************
    # NetworkTables
    # *************
    NetworkTables.initialize(server='10.30.61.2')
    sleep(.5) #give it time to initialize
    def connectionListener(connected, info):
        print(info, "; Connected=%s" % connected)

    NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)
    
    import logging
    logging.basicConfig(level=logging.INFO)
    
    rootTable = NetworkTables.getTable("VisionBox") #all data is stored in the VisionBox network table
    configTable = rootTable.getSubTable("config") #config table to edit with new config values
    outputTable = rootTable.getSubTable("output")

    tx_list = outputTable.getEntry("tx_list")
    ty_list = outputTable.getEntry("ty_list")

    def uploadConfig():
        for key,value in config.items():
            if type(value) == float or type(value) == int:
                configTable.getEntry(key).setDouble(value)
            elif type(value) == bool:
                configTable.getEntry(key).setBoolean(value)
            else:
                print("key/value not able to be converted to type!", key, value)

    #load config from file
    config = None
        
    if (not os.path.exists("./config.json")): #if the config file doesnt exist, create it
        print("config not found, creating new config")
        config = {
            "lowerHue": 90,
            "lowerSaturation": 100,
            "lowerValue": 100,
            "upperHue": 125,
            "upperSaturation": 255,
            "upperValue": 255,
            "closingIterations": 4,
            "openingIterations": 4,
            "debugMode": False
        }
        with open("./config.json","w") as f:
            json.dump(config,f)
    else:
        print("loading config from file")
        with open("./config.json","r") as f:
            config = json.load(f)
            print(config)

    #uncomment this to set config schema
    # config = {
    #     "lowerHue": 90,
    #     "lowerSaturation": 100,
    #     "lowerValue": 100,
    #     "upperHue": 125,
    #     "upperSaturation": 255,
    #     "upperValue": 255,
    #     "debugMode": False
    # }
    # with open("./config.json","w") as f:
    #     json.dump(config,f)

    uploadConfig()

    #config updates cause a change to the config dictionary and write the changes to file
    def configUpdateListener(table, key, value, isNew):
        print("valueChanged: key: '%s'; value: %s; isNew: %s" % (key, value, isNew))
        config[key] = value
        print("new config:", config)
        with open("./config.json","w") as f:
            json.dump(config,f)

    configTable.addEntryListener(configUpdateListener)

    # ****************
    # Camera Server Setup
    # ****************
    cs = cscore.CameraServer.getInstance()
    cs.enableLogging()

    camera = cscore.UsbCamera("Main",0)
    #uncomment to print video modes / settings
    for mode in camera.enumerateVideoModes():
        print("format",mode.pixelFormat,"dim",mode.width,mode.height,"fps:",mode.fps)
    for prop in camera.enumerateProperties():
        print(prop.getName(), "val", prop.get(), "default", prop.getDefault(), "opt", prop.getChoices(), "max/min", prop.getMax(), prop.getMin())
    
    camera.setConfigJson("""
    {
        "height": 720,
        "width": 1280,
        "pixel format": "mjpeg",
        "properties": [
            {
                "name": "brightness",
                "value": 70
            },
            {
                "name": "contrast",
                "value": 50
            },
            {
                "name": "saturation",
                "value": 50
            },
            {
                "name": "hue",
                "value": 50
            },
            {
                "name": "white_balance_temperature_auto",
                "value": 1
            },
            {
                "name": "power_line_frequency",
                "value": 2
            },
            {
                "name": "sharpness",
                "value": 50
            },
            {
                "name": "backlight_compensation",
                "value": 0
            },
            {
                "name": "exposure_auto",
                "value": 3
            }
        ]
    }
    """)

    #use a custom threaded cvsink solution to reduce latency
    cvSink = cscore.CvSink("Video In")
    cvSink.setSource(camera)
    cvSinkThreaded = ThreadedCvSink(cvSink).start()

    #output streams
    outputStream = cs.putVideo("MainOut", 640, 360)
    blobStream = cs.putVideo("DebugOut", 640, 360)

   # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(720, 1280, 3), dtype=np.uint8)
    print("starting main vision loop...")

    # ****************
    # Main Vision Loop
    # ****************
    while True:
      # Tell the CvSink to grab a frame from the camera and put it
      # in the source image.  If there is an error notify the output.
        time, img = cvSinkThreaded.grabFrame()
        if time == 0: 
         # Send the output the error.
            outputStream.notifyError(cvSink.getError())
         # skip the rest of the current iteration
            continue

        img = cv2.resize(img, (640,360))
        img = cv2.flip(img,-1)

        #make image into hsv format (h: 0-179, s: 0-255, v: 0-255)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv,(7,7),cv2.BORDER_DEFAULT) #blur to remove noise

        #thresholded within provided range
        mask = getHSVRange(hsv,np.array([config.get("lowerHue"),config.get("lowerSaturation"),config.get("lowerValue")],int),np.array([config.get("upperHue"),config.get("upperSaturation"),config.get("upperValue")],int))

        #opening (erode+dilate) to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        mask = cv2.erode(mask, kernel, iterations=int(config.get("openingIterations",4)))
        mask = cv2.dilate(mask, kernel, iterations=int(config.get("openingIterations",4)))

        #closing (dilate+erode) to fill holes
        mask = cv2.dilate(mask, kernel, iterations=int(config.get("closingIterations",4)))
        mask = cv2.erode(mask, kernel, iterations=int(config.get("closingIterations",4)))
        

        #blur final mask to get better results
        mask = cv2.GaussianBlur(mask,(7,7),cv2.BORDER_DEFAULT)

        #debug stream
        if config.get("debugMode",False): #if debug mode is enabled
            blobStream.putFrame(maskTest(img,mask))

        #find contours
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        circles = []
        
        for contour in contours:
            (x,y),r=cv2.minEnclosingCircle(contour)
            circularness = cv2.contourArea(contour)/(np.pi * r*r)

            if (circularness > .75):
                circles.append([x,y,r,circularness])

        circles.sort(key=lambda x: x[1],reverse=True) #sort to put the biggest contour (closest gamepiece) first
        x_s = []
        y_s = []
        for index, i in enumerate(circles):
            cv2.putText(img, str(index), (int(i[0]-10),int(i[1] + i[2]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
            cv2.line(img, (int(i[0] - i[2]/2),int(i[1] + i[2])), (int(i[0] + i[2]/2), int(i[1] + i[2])), (255,255,0), 3)
            cv2.circle(img,(int(i[0]),int(i[1])),int(i[2]),(255,255,255),2)
            tx, ty = getAnglesFromPixels(K, D, pixel=(i[0],i[1]+i[2]))
            x_s.append(tx)
            y_s.append(ty)
        tx_list.setDoubleArray(x_s)
        ty_list.setDoubleArray(y_s)
        
        outputStream.putFrame(img)

# a threaded wrapper for the cvsink class to enable faster loop times
class ThreadedCvSink:
    def __init__(self,cvSink):
        self.cvSink = cvSink
        self.stopped = False
        self.img = np.zeros(shape=(360, 640, 3), dtype=np.uint8)
        self.time = 0
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        while True:
            if self.stopped:
                return
            self.time,self.img = self.cvSink.grabFrame(self.img)
    def grabFrame(self):
        return (self.time,self.img)
    def stop(self):
        self.stopped = True

    
def getAnglesFromPixels(K,D,pixel): #K matrix; distortion coefficients; pixel: (x,y). returns angles in radians (t_x, t_y)
    #get pixel coordinates relative to center

    p_x = pixel[0] * 2
    p_y = pixel[1] * 2

    undistorted = cv2.undistortPoints(np.array([p_x,p_y], dtype=np.float64),K,D)[0][0]

    t_x = -(math.pi / 2 - math.atan2(1,undistorted[0]))
    t_y = -(math.pi / 2 - math.atan2(1,undistorted[1]))

    return (t_x,t_y)

def maskTest(src,mask):
    maskImg = cv2.bitwise_and(src,src,mask = mask)

    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(maskImg, contours, -1, (255,255,0), 3)

    for contour in contours:
        (x,y),r=cv2.minEnclosingCircle(contour)
        circularness = cv2.contourArea(contour)/(np.pi * r*r)
        cv2.putText(maskImg, str(round(circularness,2)), (int(x-15),int(y + r/4)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0),1)

    return maskImg

def getHSVRange(img, hsv_low, hsv_high): #normal, easy ranges that dont wrap the hue spectrum
    if (hsv_low[0] < hsv_high[0]):
        return cv2.inRange(img, hsv_low, hsv_high) 
    else: #range where low hue > high hue, so we want to range a spectrum that isnt continous on 0-179deg
        return cv2.inRange(img, np.array([0,hsv_low[1],hsv_low[2]]), hsv_high) + cv2.inRange(img, hsv_low, np.array([179,hsv_high[1],hsv_high[2]]))

main()
