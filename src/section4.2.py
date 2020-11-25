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

    # change flag if want to export data
    self.exportSecondaryTaskControlData = 1


    # data array to store results
    self.targetXYZPositionResults = []    
    self.closedLoopControlResults = []
    self.sinusoidAngleResults = []    
    self.secondaryTaskControlResults = []
    self.endEffectorPositions = []

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

    self.theta1 = 0.0

    # store previous angles for section 4.2
    self.prev_theta1 = 0.0
    self.prev_theta2 = 0.0
    self.prev_theta3 = 0.0
    self.prev_theta4 = 0.0

    # initialise angles for closed loop control
    self.t1 = 0.0
    self.t2 = 0.0
    self.t3 = 0.0
    self.t4 = 0.0
    

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

    self.orangeYPos = rospy.Subscriber("/orangeYPosEst", Float64, self.getOrangeYPos)
    self.orangeYPosData = Float64() 

    # end effector position using cv
    self.endEffYPos = rospy.Subscriber("/endEffYPos", Float64, self.getEndEffY)
    self.endEffYPosData = Float64() 

    # initialize a publisher to send joints' angular position to the robot
    self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10) 
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10) 

    self.time_previous_step = np.array([rospy.get_time()], dtype='float64') 
 
    # initialise error for secondary task control task
    self.secondary_error = np.array([0.0,0.0, 0.0], dtype='float64')  
    self.secondary_error_d = np.array([0.0,0.0, 0.0], dtype='float64')

    # initial distance between end effector and orange square
    self.prev_distance = None

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

  def cacheOrangeSquarePos(self, pos):
    if len(self.orangeSquareCache) < 1000:
      self.orangeSquareCache.append(pos)
    else:
      self.orangeSquareCache = []
      self.orangeSquareCache.append(pos)


  def getTargetYPos(self, data):
    try:
      self.targetYPosData = data.data
    except CvBridgeError as e:
      print(e)      

  def getOrangeYPos(self, data):
    try:
      self.orangeYPosData = data.data
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
      self.cacheObjectPos(keypoints[0].pt)
      return keypoints[0].pt
    else:
      return self.objectCache[-1]

  def get_orange_square_coordinates(self, image):
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
    params.minCircularity = 0.00
    params.maxCircularity = 0.82

    detector = cv2.SimpleBlobDetector_create(params)

    # convert black pixels to white and object to black
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    res[thresh == 0] = 255
    res = cv2.bitwise_not(thresh)

    # detect orange square
    keypoints = detector.detect(res)

    # if x < 420.5, out of range of square and circle detected. Then return previous correct square position
    if keypoints:
      idx = 0
      if len(keypoints) == 2:
        if  keypoints[0].pt[0] < 420.0:
          idx = 1
      elif len(keypoints) == 1:
        if keypoints[0].pt[0] < 420.0:
          return self.orangeSquareCache[-1]
      self.cacheOrangeSquarePos(keypoints[idx].pt)
      return keypoints[idx].pt
    else:
      return self.orangeSquareCache[-1]      

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
   

  def getJacobian(self, theta1, theta2, theta3, theta4):
    # this expression is common in many derivatives
    commonExpression = 3.5 + 3*np.cos(theta4)

    # calculate derivatives with respect to joint angles
    dxd1 = np.cos(theta1)*np.sin(theta3)*(commonExpression) + np.sin(theta1)*np.sin(theta2)*np.cos(theta3)*(commonExpression) + 3*np.sin(theta1)*np.cos(theta2)*np.sin(theta4)
    dxd2 = 3*np.sin(theta2)*np.cos(theta1)*np.sin(theta4) - np.cos(theta1)*np.cos(theta2)*np.cos(theta3)*(commonExpression)    
    dxd3 = np.cos(theta3)*np.sin(theta1)*(commonExpression) + np.sin(theta3)*np.cos(theta1)*np.sin(theta2)*(commonExpression)    
    dxd4 = -3*np.cos(theta4)*np.cos(theta1)*np.cos(theta2) + 3*np.sin(theta4)*np.cos(theta1)*np.sin(theta2)*np.cos(theta3) - 3*np.sin(theta4)*np.sin(theta1)*np.sin(theta3)    
    dyd1 = -np.cos(theta1)*np.sin(theta2)*np.cos(theta3)*(commonExpression) + np.sin(theta1)*np.sin(theta3)*(commonExpression) - 3*np.cos(theta1)*np.cos(theta2)*np.sin(theta4)    
    dyd2 = 3*np.sin(theta2)*np.sin(theta1)*np.sin(theta4) - np.cos(theta2)*np.sin(theta1)*np.cos(theta3)*(commonExpression)    
    dyd3 = np.sin(theta3)*np.sin(theta1)*np.sin(theta2)*(commonExpression) - np.cos(theta3)*np.cos(theta1)*(commonExpression)    
    dyd4 = -3*np.cos(theta4)*np.sin(theta1)*np.cos(theta2) + 3*np.sin(theta4)*np.sin(theta1)*np.sin(theta2)*np.cos(theta3) + 3*np.sin(theta4)*np.cos(theta1)*np.sin(theta3)     
    dzd1 = 0.0  
    dzd2 = -np.sin(theta2)*np.cos(theta3)*(commonExpression) - 3*np.cos(theta2)*np.sin(theta4)
    dzd3 = -np.sin(theta3)*np.cos(theta2)*(commonExpression)
    dzd4 = -3*np.sin(theta4)*np.cos(theta2)*np.cos(theta3) - 3*np.cos(theta4)*np.sin(theta2)    

    # combine derivatives into jacobian array
    jacobian = np.array([
      [dxd1, dxd2, dxd3, dxd4], 
      [dyd1, dyd2, dyd3, dyd4], 
      [dzd1, dzd2, dzd3, dzd4]])

    return jacobian


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

  # def getTheta1(X, Y, theta2, theta3, theta4):
    

  def getTheta2And4(self, image):
    # get joint positions
    try:
      joint2Pos = self.detect_blue(image)
      self.cacheBlueCirclePos(joint2Pos)
    except:
      joint2Pos = self.blueCircleCache[-1]
    joint3Pos = joint2Pos
    joint3Pos = joint3Pos
    
    try:
      joint4Pos =  self.detect_green(image)
      self.cacheGreenCirclePos(joint4Pos)
    except:
      joint4Pos = self.greenCircleCache[-1]

    # get theta2
    theta2 = np.arctan2(joint3Pos[0]- joint4Pos[0], joint3Pos[1] - joint4Pos[1])

    try:
      endEffectorPos = self.detect_red(image)
      self.cacheRedCirclePos(endEffectorPos)
    except:
      endEffectorPos = self.redCircleCache[-1]

    # get theta4
    theta4 = np.arctan2(joint4Pos[0]- endEffectorPos[0], joint4Pos[1] - endEffectorPos[1]) - theta2

    return theta2, theta4

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

  def getOrangeSquareCoordinates(self, image):  
    # get position of circular object
    try:
      orangeSquarePos = self.get_orange_square_coordinates(image)
      self.cacheObjectPos(orangeSquarePos)
    except:
      orangeSquarePos = self.objectCache[-1]
    # position of first joint
    try:
      joint1Pos = self.detect_yellow(image)
      self.cacheYellowCirclePos(joint1Pos)
    except:
      joint1Pos = self.yellowCircleCache[-1]

    # calculate distance from base to object
    # x, y, z of target is based on base frame as defined through D-H
    distBaseToObjectPixels = np.sum((joint1Pos - orangeSquarePos)**2)
    distBaseToObjectMeters = self.meterPerPixel * np.sqrt(distBaseToObjectPixels)    
    baseToTargetAngle = np.arctan2(joint1Pos[0]- orangeSquarePos[0], joint1Pos[1] - orangeSquarePos[1])
    orangeSquareZ = distBaseToObjectMeters*np.cos(baseToTargetAngle)
    orangeSquareX = distBaseToObjectMeters*np.sin(baseToTargetAngle)*-1   

    return orangeSquareX, orangeSquareZ  

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


  def publishJointAngles(self, theta1, theta2, theta3, theta4):

    # adjust joint angles
    self.joint1=Float64()
    self.joint1.data = theta1
    self.joint2=Float64()
    self.joint2.data = theta2
    self.joint3=Float64()
    self.joint3.data = theta3
    self.joint4=Float64()
    self.joint4.data = theta4   

    # Publish the results
    try:
      rate = rospy.Rate(12)
      rate.sleep()
      self.robot_joint1_pub.publish(self.joint1)
      self.robot_joint2_pub.publish(self.joint2)
      self.robot_joint3_pub.publish(self.joint3)
      self.robot_joint4_pub.publish(self.joint4)
    except CvBridgeError as e:
      print(e)   


  def getEndEffectorToSquareDistance(self, endEffX, endEffY, endEffZ, orangeX, orangeY, orangeZ):
    endEffectorPos = np.array([endEffX, endEffY, endEffZ])
    squarePos = np.array([orangeX, orangeY, orangeZ])
    distance = np.sqrt(np.sum((endEffectorPos-squarePos)**2))
    return distance

  def controlWithSecondaryTask(self,
    endEffX,
    endEffY,
    endEffZ,
    targetX, 
    targetY, 
    targetZ,
    orangeX, 
    orangeY, 
    orangeZ):

    # get current time step and calculate dt
    cur_time = np.array([rospy.get_time()])
    dt = cur_time - self.time_previous_step
    self.time_previous_step = cur_time

    # P and Dgain
    P = 0.29
    D = 0.1
    I = 0.05
    K_d = np.array([[D, 0, 0],[0, D, 0], [0, 0, D]])   
    K_p = np.array([[P, 0, 0],[0, P, 0], [0, 0, P]])

    # set joint angle values
    theta1 = self.t1
    theta2 = self.t2
    theta3 = self.t3
    theta4 = self.t4  

    # get dq
    dq1 = theta1 - self.prev_theta1
    if dq1 < 0.1:
      dq1 = 0.1
    dq2 = theta2 - self.prev_theta2
    if dq2 < 0.1:
      dq2 = 0.1    
    dq3 = theta3 - self.prev_theta3
    if dq3 < 0.1:
      dq3 = 0.1    
    dq4 = theta4 - self.prev_theta4
    if dq4 < 0.1:
      dq4 = 0.1  
   
    self.prev_theta1 = theta1
    self.prev_theta2 = theta2
    self.prev_theta3 = theta3
    self.prev_theta4 = theta4

    # distance between end effector and orange square
    distance = self.getEndEffectorToSquareDistance(endEffX, endEffY, endEffZ, orangeX, orangeY, orangeZ)
    distance_diff = distance - self.prev_distance
    self.prev_distance = distance

    # calculate derivative of cost with respect to joint angles
    dw_dq = np.array([distance_diff/dq1, distance_diff/dq2, distance_diff/dq3, distance_diff/dq4])

    # get end effector and target pos
    endEffectorPosition = np.array([endEffX, endEffY, endEffZ])
    targetPos = np.array([targetX, targetY, targetZ])

    # estimate derivative of error
    self.secondary_error_d = ((targetPos - endEffectorPosition) - self.secondary_error)/dt
    self.secondary_error = targetPos - endEffectorPosition   

    # get jacobian and calculate jacobian pseudoinverse
    J = self.getJacobian(theta1, theta2, theta3, theta4)
    J_inv = np.linalg.pinv(J)
    J_pseudo_inv = np.dot(J.transpose(), np.linalg.pinv((np.dot(J, J.transpose()))))

    q = np.array([theta1, theta2, theta3, theta4])
    dq_d = np.dot(J_pseudo_inv, np.dot(K_d, self.secondary_error_d) + np.dot(K_p, self.secondary_error)  ) + np.dot((np.eye(4) - np.dot(J_pseudo_inv, J)), I*dw_dq  )
    q_d  = q + (dt * dq_d)

    # export results
    if self.exportSecondaryTaskControlData == 1:
      self.secondaryTaskControlResults.append([
        rospy.get_time(), 
        targetX, 
        endEffectorPosition[0],
        targetY,
        endEffectorPosition[1],
        targetZ,
        endEffectorPosition[2],
        orangeX, 
        orangeY, 
        orangeZ])
      secondaryTaskControlResultsDF = pd.DataFrame(
        self.secondaryTaskControlResults, 
        columns=['time', 'targetX', 'endEffectorX', 'targetY', 'endEffectorY', 'targetZ', 'endEffectorZ',
        'orangeX', 
        'orangeY', 
        'orangeZ'])
      secondaryTaskControlResultsDF.to_csv(os.getcwd() + '/src/ivr_assignment/exports/secondaryTaskControlResults.csv')

    # publish angles
    self.publishJointAngles(q_d[0], q_d[1], q_d[2], q_d[3])

    # new joint angle values
    self.t1 = q_d[0]
    self.t2 = q_d[1]
    self.t3 = q_d[2]
    self.t4 = q_d[3]


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


    targetX, targetZ = self.getObjectCoordinates(self.cv_image1)
    # target Y position:
    targetY = self.targetYPosData    

    # end effector position as calculated using cv:
    endEffX, endEffZ = self.getEndEffectorCoordinates(self.cv_image1)
    endEffY = self.endEffYPosData

    # SECTION 4.2

    # get orange square coordinates
    orangeSquareX, orangeSquareZ = self.getOrangeSquareCoordinates(self.cv_image1)
    orangeSquareY = self.orangeYPosData

    # set distance from end effector to orange square
    if not self.prev_distance:
      self.prev_distance = self.getEndEffectorToSquareDistance(endEffX, endEffY, endEffZ, orangeSquareX, orangeSquareY, orangeSquareZ)

    # perform control
    self.controlWithSecondaryTask(
        endEffX,
        endEffY,
        endEffZ,
        targetX, 
        targetY, 
        targetZ,
        orangeSquareX, 
        orangeSquareY, 
        orangeSquareZ)



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


