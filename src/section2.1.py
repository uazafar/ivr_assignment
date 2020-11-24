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
    self.exportSinusoidAngles = 0

    # data array to store results
    self.sinusoidAngleResults = []    


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

    # joint 3 angle and target Y pos require running image2.py
    self.jointAngle3 = rospy.Subscriber("/jointAngle3", Float64, self.getJointAngle3)
    self.jointAngle3Data = Float64()

    # initialize a publisher to send joints' angular position to the robot
    self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10) 
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10) 

    # set up publisher   
    # rospy.init_node('publisher_node',anonymous=True)
    self.jointAngle2 = rospy.Publisher("jointAngle2", Float64, queue_size=10)
    self.jointAngle4 = rospy.Publisher("jointAngle4", Float64, queue_size=10)
    self.actualJointAngle2 = rospy.Publisher("actualJointAngle2", Float64, queue_size=10)
    self.actualJointAngle3 = rospy.Publisher("actualJointAngle3", Float64, queue_size=10)
    self.actualJointAngle4 = rospy.Publisher("actualJointAngle4", Float64, queue_size=10)
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



  def exportDetectedSinusoidAngles(self, theta2, inputAngle2, theta3, inputAngle3, theta4, inputAngle4):
    if self.exportSinusoidAngles == 1:
      self.sinusoidAngleResults.append([
        rospy.get_time(),
        inputAngle2, theta2, inputAngle3, theta3, inputAngle4, theta4])
      sinusoidAngleResultsDF = pd.DataFrame(
        self.sinusoidAngleResults, 
        columns=['time', 'inputAngle2', 'theta2', 'inputAngle3', 'theta3', 'inputAngle4', 'theta4'])
      sinusoidAngleResultsDF.to_csv(os.getcwd() + '/src/ivr_assignment/exports/detectSinusoidAngles.csv')


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



    # SECTION 2.1
    # adjust joint angles using sinusoidal signals
    self.joint2=Float64()
    inputAngle2 = (np.pi/2) * np.sin((np.pi/15) * rospy.get_time())
    self.joint2.data = inputAngle2
    self.joint3=Float64()
    inputAngle3 = (np.pi/2) * np.sin((np.pi/18) * rospy.get_time())
    self.joint3.data = inputAngle3    
    self.joint4=Float64()
    inputAngle4 = (np.pi/3) * np.sin((np.pi/20) * rospy.get_time())
    self.joint4.data = inputAngle4      
    # Publish the results
    try:
      self.robot_joint2_pub.publish(self.joint2)
      self.robot_joint3_pub.publish(self.joint3)
      self.robot_joint4_pub.publish(self.joint4)
    except CvBridgeError as e:
      print(e)

    # stimate angles for joint 2 and 4
    theta2, theta4 = self.getTheta2And4(self.cv_image1)
    # get joint angle 3 (which is sent to a topic by running image2.py)
    try:
      theta3 = float(self.jointAngle3Data)
    # on first iteration, theta3 sometimes unavailable from topic but value is 0.0 at beginning
    except:
      theta3 = 0.0

    # publish detected joint angles
    self.package = Float64()
    self.package.data = theta2
    self.jointAngle2.publish(self.package)
    self.package = Float64()
    self.package.data = theta4
    self.jointAngle4.publish(self.package)

    # export input and detected angles when joints modulated with sinusoids 
    if self.exportSinusoidAngles == 1:
      self.exportDetectedSinusoidAngles(theta2, inputAngle2, theta3, inputAngle3, theta4, inputAngle4)    


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


