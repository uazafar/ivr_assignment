#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import message_filters


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):

    # define flag to determine whether joints should be modulated using sinusoids
    self.modulateJointsWithSinusoids = 1

    self.meterPerPixel = None

    # define a cache to store positions of circles
    self.greenCircleCache = []
    self.redCircleCache = []
    self.blueCircleCache = []
    self.yellowCircleCache = []
    self.objectCache = []

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
    self.jointAngle3 = rospy.Subscriber("/jointAngle3", Float64, self.getJointAngle3)
    self.jointAngle3Data = Float64()

    # define variables to store robot joints
    self.jointAngle3 = rospy.Subscriber("/robot/joint1_position_controller/command", Float64, self.getRobotJoint1State)
    self.jointAngle3 = rospy.Subscriber("/robot/joint2_position_controller/command", Float64, self.getRobotJoint2State)
    self.jointAngle3 = rospy.Subscriber("/robot/joint3_position_controller/command", Float64, self.getRobotJoint3State)
    self.jointAngle3 = rospy.Subscriber("/robot/joint4_position_controller/command", Float64, self.getRobotJoint4State)
    self.robotJoint1State = Float64()
    self.robotJoint2State = Float64()
    self.robotJoint3State = Float64()
    self.robotJoint4State = Float64()


    # initialize a publisher to send joints' angular position to the robot
    # self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    # self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    # self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10) 
    # self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10) 

    # set up publisher   
    # rospy.init_node('publisher_node',anonymous=True)
    # self.jointAngle2 = rospy.Publisher("jointAngle2", Float64, queue_size=10)
    # self.jointAngle4 = rospy.Publisher("jointAngle4", Float64, queue_size=10)
    # self.actualJointAngle2 = rospy.Publisher("actualJointAngle2", Float64, queue_size=10)
    # self.actualJointAngle4 = rospy.Publisher("actualJointAngle4", Float64, queue_size=10)
    # self.targetZPosEst = rospy.Publisher("targetZPosEst", Float64, queue_size=10)
    # self.targetXPosEst = rospy.Publisher("targetXPosEst", Float64, queue_size=10)
    self.rate = rospy.Rate(10) #hz
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

  def getJointAngle3(self, data):
    try:
      self.jointAngle3Data = data.data
    except CvBridgeError as e:
      print(e)


  def getRobotJoint1State(self, data):
    try:
      self.robotJoint1State = data.data 
    except CvBridgeError as e:
      print(e)

  def getRobotJoint2State(self, data):
    try:
      self.robotJoint2State = data.data 
    except CvBridgeError as e:
      print(e)

  def getRobotJoint3State(self, data):
    try:
      self.robotJoint3State = data.data 
    except CvBridgeError as e:
      print(e)

  def getRobotJoint4State(self, data):
    try:
      self.robotJoint4State = data.data 
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
      # return 0,0 if object cannot be detected
      return self.objectCache[-1]

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

  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):

    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # calculate metes per pixel and store value
    if not self.meterPerPixel:
      self.meterPerPixel = self.pixel2meterYellowToBlue(self.cv_image1)      
    
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)

    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
    except CvBridgeError as e:
      print(e)



    # get target distance

    inputAngle3 = self.jointAngle3Data 



    # SECTION 3.1

    # ensure robot position is correct when all angles zero
    # endEffectorStraight = self.getEndEffectorXYZ(0,0,0,0)
    # assert endEffectorStraight[0] < 1.0e-10
    # assert endEffectorStraight[1] < 1.0e-10
    # assert endEffectorStraight[2] == 9.0

    # print(endEffectorStraight)


    # SECTION 3.2

    # theta1, theta2, theta3, theta4 = 0.0, 0.0, 0.0, 0.0
    # jacobian = self.getJacobian(theta1, theta2, theta3, theta4)



    # random 

    im2=cv2.imshow('window2', self.cv_image1)
    cv2.waitKey(1)


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


