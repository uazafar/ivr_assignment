#!/usr/bin/env python3

import roslib
import os
import sys
import rospy
import cv2
import numpy as np
import pandas as pd
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import time


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):

    self.exportEstimatedTargetPositionData = 0

    # data array to store results
    self.targetXYZPositionResults = []    

    self.meterPerPixel = None

    # define a cache to store positions of circles
    self.greenCircleCache = []
    self.redCircleCache = []
    self.blueCircleCache = []
    self.yellowCircleCache = []
    # if object not visible straight away, initialise with the following position (behind robot)
    self.objectCache = [[400,400]]
    self.orangeSquareCache = [[500,354]]

    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()

    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)

    # target Y pos require running image2.py
    self.targetYPos = rospy.Subscriber("/targetYPosEst", Float64, self.getTargetYPos)
    self.targetYPosData = Float64() 

    # initialize a publisher to send joints' angular position to the robot
    self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10) 
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10) 


    # set up publisher   
    # rospy.init_node('publisher_node',anonymous=True)
    self.targetZPosEst = rospy.Publisher("targetZPosEst", Float64, queue_size=10)
    self.targetXPosEst = rospy.Publisher("targetXPosEst", Float64, queue_size=10)
    self.rate = rospy.Rate(1) #hz
    self.time = rospy.get_time()


  def cacheBlueCirclePos(self, pos):
    if len(self.blueCircleCache) < 1000:
      self.blueCircleCache.append(pos)
    else:
      self.blueCircleCache = []
      self.blueCircleCache.append(pos)

  def cacheYellowCirclePos(self, pos):
    if len(self.yellowCircleCache) < 1000:
      self.yellowCircleCache.append(pos)
    else:
      self.yellowCircleCache = []
      self.yellowCircleCache.append(pos)

  def cacheGreenCirclePos(self, pos):
    if len(self.greenCircleCache) < 1000:
      self.greenCircleCache.append(pos)
    else:
      self.greenCircleCache = []
      self.greenCircleCache.append(pos)

  def cacheRedCirclePos(self, pos):
    if len(self.redCircleCache) < 1000:
      self.redCircleCache.append(pos)
    else:
      self.redCircleCache = []
      self.redCircleCache.append(pos)


  def cacheObjectPos(self, pos):
    if len(self.objectCache) < 1000:
      self.objectCache.append(pos)
    else:
      self.objectCache = []
      self.objectCache.append(pos)

  def getTargetYPos(self, data):
    try:
      self.targetYPosData = data.data
    except CvBridgeError as e:
      print(e)      


  # In this method you can focus on detecting the centre of the red circle
  def detect_red(self,image):
      # Isolate the blue colour in the image as a binary image
      mask = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
      # This applies a dilate that makes the binary region larger (the more iterations the larger it becomes)
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      # Obtain the moments of the binary image
      M = cv2.moments(mask)
      # Calculate pixel coordinates for the centre of the blob
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return np.array([cx, cy])
 

  # Detecting the centre of the green circle
  def detect_green(self,image):
      mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return np.array([cx, cy])


  # Detecting the centre of the blue circle
  def detect_blue(self,image):
      mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return np.array([cx, cy])

  # Detecting the centre of the blue circle
  def detect_orange(self,image):
      mask = cv2.inRange(image, (0, 100, 200), (0, 200, 255))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return np.array([cx, cy])

  # Detecting the centre of the yellow circle
  def detect_yellow(self,image):
      mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      return np.array([cx, cy])


  # Calculate the conversion from pixel to meter
  def pixel2meterYellowToRed(self,image):
      # Obtain the centre of each coloured blob
      try:
        circle1Pos = self.detect_yellow(image)
        self.cacheYellowCirclePos(circle1Pos)
      except:
        circle1Pos = self.yellowCircleCache[-1]

      try:
        circle2Pos = self.detect_red(image)
        self.cacheRedCirclePos(circle2Pos)
      except:
        circle2Pos = self.redCircleCache[-1]

      # find the distance between two circles
      dist = np.sum((circle1Pos - circle2Pos)**2)
      return 9 / np.sqrt(dist)


  # Calculate the conversion from pixel to meter
  def pixel2meterYellowToBlue(self,image):
      # Obtain the centre of each coloured blob
      try:
        circle1Pos = self.detect_yellow(image)
        self.cacheYellowCirclePos(circle1Pos)
      except:
        circle1Pos = self.yellowCircleCache[-1]

      try:
        circle2Pos = self.detect_blue(image)
        self.cacheBlueCirclePos(circle2Pos)
      except:
        circle2Pos = self.blueCircleCache[-1]

      # find the distance between two circles
      dist = np.sum((circle1Pos - circle2Pos)**2)
      return 2.5 / np.sqrt(dist)

  # Calculate the conversion from pixel to meter
  def pixel2meterBlueToGreen(self,image):

      try:
        circle1Pos = self.detect_blue(image)
        self.cacheBlueCirclePos(circle1Pos)
      except:
        circle1Pos = self.blueCircleCache[-1]

      try:
        circle2Pos = self.detect_green(image)
        self.cacheGreenCirclePos(circle2Pos)
      except:
        circle2Pos = self.greenCircleCache[-1]        

      # find the distance between two circles
      dist = np.sum((circle1Pos - circle2Pos)**2)
      return 3.5 / np.sqrt(dist)

  # Calculate the conversion from pixel to meter
  def pixel2meterGreenToRed(self,image):

      try:
        circle1Pos = self.detect_green(image)
        self.cacheGreenCirclePos(circle1Pos)
      except:
        circle1Pos = self.greenCircleCache[-1] 

      try:
        circle2Pos = self.detect_red(image)
        self.cacheRedCirclePos(circle2Pos)
      except:
        circle2Pos = self.redCircleCache[-1]  

      # find the distance between two circles
      dist = np.sum((circle1Pos - circle2Pos)**2)
      return 3.0 / np.sqrt(dist)


  def get_object_coordinates(self, image):
    # Threshold the HSV image to get only orange colors (of object)
    mask = cv2.inRange(image, (0,20,100), (40,100,150))
    res = cv2.bitwise_and(image, image, mask= mask)

    # convert image to greyscale
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # create parameters for blob detector to detect circles
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = False
    params.filterByInertia = 1.0
    params.filterByConvexity = False
    params.filterByCircularity = 1.0
    params.minCircularity = 0.87
    params.maxCircularity = 1.00
    detector = cv2.SimpleBlobDetector_create(params)

    # convert black pixels to white and object to black
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    res[thresh == 0] = 255
    res = cv2.bitwise_not(thresh)

    # detect circles
    keypoints = detector.detect(res)
    if keypoints:
      # self.cacheObjectPos(keypoints[0].pt)
      return keypoints[0].pt
    else:
      return self.objectCache[-1]


  def getObjectCoordinates(self, image):  
    # get position of circular object
    try:
      objectPos = self.get_object_coordinates(image)
      self.cacheObjectPos(objectPos)
    except:
      objectPos = self.objectCache[-1]
    # position of first joint
    try:
      joint1Pos = self.detect_yellow(image)
      self.cacheYellowCirclePos(joint1Pos)
    except:
      joint1Pos = self.yellowCircleCache[-1]

    # calculate distance from base to object
    # x, y, z of target is based on base frame as defined through D-H
    distBaseToObjectPixels = np.sum((joint1Pos - objectPos)**2)
    distBaseToObjectMeters = self.meterPerPixel * np.sqrt(distBaseToObjectPixels)    
    baseToTargetAngle = np.arctan2(joint1Pos[0]- objectPos[0], joint1Pos[1] - objectPos[1])
    targetZ = distBaseToObjectMeters*np.cos(baseToTargetAngle)
    targetX = distBaseToObjectMeters*np.sin(baseToTargetAngle)*-1   
    
    return targetX, targetZ 


  def exportEstimatedTargetPosition(self, targetX, targetY, targetZ):
    if self.exportTargetPosition == 1:
      self.targetXYZPositionResults.append([
        rospy.get_time(),
        targetX, targetY, targetZ])
      targetXYZPositionResultsDF = pd.DataFrame(
        self.targetXYZPositionResults, 
        columns=['time', 'targetX', 'targetY', 'targetZ'])
      targetXYZPositionResultsDF.to_csv(os.getcwd() + '/src/ivr_assignment/exports/targetPosition.csv')


  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):

    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
    except CvBridgeError as e:
      print(e)      

    # calculate meters per pixel and store value
    if not self.meterPerPixel:
      self.meterPerPixel = self.pixel2meterYellowToRed(self.cv_image1)      
    
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)


    #SECTION 2.2:

    targetX, targetZ = self.getObjectCoordinates(self.cv_image1)
    # target Y position:
    targetY = self.targetYPosData 

    # export data
    if self.exportEstimatedTargetPositionData == 1:
      self.exportEstimatedTargetPosition(targetX, targetY, targetZ)


    # publish estimated position of target
    self.package = Float64()
    self.package.data = targetZ
    self.targetZPosEst.publish(self.package)
    self.package = Float64()
    self.package.data = targetX
    self.targetXPosEst.publish(self.package)



    # im2=cv2.imshow('window2', self.cv_image1)
    # cv2.waitKey(1)


# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)


