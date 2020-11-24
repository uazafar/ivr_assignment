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


    self.meterPerPixel = None

    # define a cache to store positions of circles
    self.greenCircleCache = []
    self.redCircleCache = []
    self.blueCircleCache = []
    self.yellowCircleCache = []
    # if object not visible straight away, initialise with the following position (behind robot)
    self.objectCache = [[400,400]]
    self.orangeSquareCache = [[500,354]]

    # Define D-H variables
    self.d1, self.d2, self.d3, self.d4 = 2.5, 0.0, 0.0, 0.0
    self.a1, self.a2, self.a3, self.a4 = 0.0, 0.0, -3.5, -3.0
    self.alpha1, self.alpha2, self.alpha3, self.alpha4 = np.pi/2, -np.pi/2, np.pi/2, 0.0

    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()

    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)

    # joint 3 angle and target Y pos require running image2.py
    self.jointAngle3 = rospy.Subscriber("/jointAngle3", Float64, self.getJointAngle3)
    self.jointAngle3Data = Float64()
    self.targetYPos = rospy.Subscriber("/targetYPosEst", Float64, self.getTargetYPos)
    self.targetYPosData = Float64() 
    self.endEffYPos = rospy.Subscriber("/endEffYPos", Float64, self.getEndEffY)
    self.endEffYPosData = Float64()     

    # position of end effector as calculated using cv
    self.endEffXPos = rospy.Publisher("endEffXPos", Float64, queue_size=10) 
    self.endEffZPos = rospy.Publisher("endEffZPos", Float64, queue_size=10)    

    # initialize a publisher to send joints' angular position to the robot
    self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10) 
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10) 


    # set up publisher   
    # rospy.init_node('publisher_node',anonymous=True)
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

  def cacheOrangeSquarePos(self, pos):
    if len(self.orangeSquareCache) < 1000:
      self.orangeSquareCache.append(pos)
    else:
      self.orangeSquareCache = []
      self.orangeSquareCache.append(pos)

  def getJointAngle3(self, data):
    try:
      self.jointAngle3Data = data.data
    except CvBridgeError as e:
      print(e)

  def getTargetYPos(self, data):
    try:
      self.targetYPosData = data.data
    except CvBridgeError as e:
      print(e)      


  def getEndEffY(self, data):
    try:
      self.endEffYPosData = data.data
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



  # A function to transform from one frame to another using 4 D-H parameters
  def transform(self, theta, d, a, alpha):
    rZ = np.array([
      [np.cos(theta), -np.sin(theta), 0, 0], 
      [np.sin(theta), np.cos(theta), 0, 0], 
      [0, 0, 1, 0], 
      [0, 0, 0, 1]])  

    dZ = np.array([
      [1, 0, 0, 0], 
      [0, 1, 0, 0], 
      [0, 0, 1, d], 
      [0, 0, 0, 1]])  

    dA = np.array([
      [1, 0, 0, a], 
      [0, 1, 0, 0], 
      [0, 0, 1, 0], 
      [0, 0, 0, 1]])  

    rX = np.array([
      [1, 0, 0, 0], 
      [0, np.cos(alpha), -np.sin(alpha), 0], 
      [0, np.sin(alpha), np.cos(alpha), 0], 
      [0, 0, 0, 1]])  

    return rZ @ dZ @ dA @ rX
   

  # get end effector positon using FK
  def getEndEffectorXYZ(self, theta1, theta2, theta3, theta4):
    # end effector matrix
    endEffectorPosInBaseMatrix = self.getEndEffectorToBaseFrameMatrix(
    theta1, theta2, theta3, theta4)

    endEffectorPos = np.array([
      round(endEffectorPosInBaseMatrix[0:3][0][3], 3) , 
      round(endEffectorPosInBaseMatrix[0:3][1][3], 3) , 
      round(endEffectorPosInBaseMatrix[0:3][2][3], 3)]) 

    return endEffectorPos   

  def getEndEffectorToBaseFrameMatrix(self,
    theta1, theta2, theta3, theta4):
    return self.transform(theta1, self.d1, self.a1, self.alpha1) @ \
      self.transform(theta2 - np.pi/2, self.d2, self.a2, self.alpha2) @ \
      self.transform(theta3, self.d3, self.a3, self.alpha3) @ \
      self.transform(theta4, self.d4, self.a4, self.alpha4)


  # this function calculates enf effector position using cv (not FK)
  def getEndEffectorCoordinates(self, image):  
    # get position red circle
    try:
      endEffPos = self.detect_red(image)
      self.cacheRedCirclePos(endEffPos)
    except:
      endEffPos = self.redCircleCache[-1]
    # position of first joint
    try:
      joint1Pos = self.detect_yellow(image)
      self.cacheYellowCirclePos(joint1Pos)
    except:
      joint1Pos = self.yellowCircleCache[-1]

    # calculate distance from base to object
    # x, y, z of target is based on base frame as defined through D-H
    distBaseToEEPixels = np.sum((joint1Pos - endEffPos)**2)
    distBaseToEEMeters = self.meterPerPixel * np.sqrt(distBaseToEEPixels)    
    baseToEEAngle = np.arctan2(joint1Pos[0]- endEffPos[0], joint1Pos[1] - endEffPos[1])
    endEffZ = distBaseToEEMeters*np.cos(baseToEEAngle)
    endEffX = distBaseToEEMeters*np.sin(baseToEEAngle)*-1   

    return endEffX, endEffZ 


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



    # SECTION 3.1

    # robot position when all angles zero
    # endEffectorStraight = self.getEndEffectorXYZ(0,0,0,0)

    # end effector position as calculated using cv:
    endEffX, endEffZ = self.getEndEffectorCoordinates(self.cv_image1)
    endEffY = self.endEffYPosData

    self.package = Float64()
    self.package.data = endEffX
    self.endEffXPos.publish(self.package)
    self.package = Float64()
    self.package.data = endEffZ
    self.endEffZPos.publish(self.package)

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


