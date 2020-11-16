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

    # define a cache to store positions of circles
    self.greenCircleCache = []
    self.redCircleCache = []
    self.blueCircleCache = []
    self.yellowCircleCache = []
    self.objectCache = []

    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    self.jointAngle3 = rospy.Subscriber("/jointAngle3", Float64, self.getJointAngle3)
    self.jointAngle3Data = Float64()

    # self.image_sub1 = message_filters.Subscriber("/camera1/robot/image_raw",Image)
    # self.jointAngle3 = message_filters.Subscriber("/jointAngle3",Float64)
    # self.ts = message_filters.TimeSynchronizer([self.image_sub1, self.jointAngle3], 10)
    # self.ts.registerCallback(self.callback1)

    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()

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
    self.actualJointAngle4 = rospy.Publisher("actualJointAngle4", Float64, queue_size=10)
    self.targetZPosEst = rospy.Publisher("targetZPosEst", Float64, queue_size=10)
    self.targetYPosEst = rospy.Publisher("targetYPosEst", Float64, queue_size=10)
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
  def pixel2meter(self,image):
      # Obtain the centre of each coloured blob
      circle1Pos = self.detect_blue(image)
      circle2Pos = self.detect_green(image)
      # find the distance between two circles
      dist = np.sum((circle1Pos - circle2Pos)**2)
      return 3.5 / np.sqrt(dist)

  # Calculate the conversion from pixel to meter
  def pixel2meterEndEff(self,image):
      # Obtain the centre of each coloured blob
      circle1Pos = self.detect_green(image)
      circle2Pos = self.detect_red(image)
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


  def get_distance_base_to_object(self, joint1Pos, joint2Pos, objectPos):
    if objectPos[0] > joint2Pos[0] and objectPos[0] > joint1Pos[0] and objectPos[1] < joint1Pos[1] and objectPos[1] < joint2Pos[1]:
      # theta1 = np.arctan2(objectPos[0] - joint1Pos[0], objectPos[1] - joint1Pos[1])
      theta1 = np.arctan2(objectPos[0] - joint1Pos[0], joint1Pos[1] - objectPos[1])
      # theta2 = 1 - np.arctan2(objectPos[0] - joint2Pos[0], objectPos[1] - joint2Pos[1])
      theta2 = np.pi - np.arctan2(objectPos[0] - joint2Pos[0], joint2Pos[1] - objectPos[1])
      theta3 = np.pi - theta1 - theta2
      distJoint1ToObject = (2.5 * np.sin(theta2))/np.sin(theta3)
    elif objectPos[0] < joint2Pos[0] and objectPos[0] < joint1Pos[0] and objectPos[1] < joint1Pos[1] and objectPos[1] > joint2Pos[1]:
      # theta1 = np.abs(np.arctan2(objectPos[0] - joint1Pos[0], objectPos[1] - joint1Pos[1]))
      theta1 = np.abs(np.arctan2(objectPos[0] - joint1Pos[0], joint1Pos[1] - objectPos[1]))
      # theta2 = 1 - np.abs(np.arctan2(objectPos[0] - joint2Pos[0], objectPos[1] - joint2Pos[1]))
      theta2 = np.pi - np.abs(np.arctan2(objectPos[0] - joint2Pos[0], joint2Pos[1] - objectPos[1]))
      theta3 = np.pi - theta1 - theta2
      distJoint1ToObject = (2.5 * np.sin(theta2))/np.sin(theta3)
    elif objectPos[0] > joint2Pos[0] and objectPos[0] > joint1Pos[0] and objectPos[1] < joint1Pos[1] and objectPos[1] < joint2Pos[1]:
      # theta1 = np.arctan2(objectPos[0] - joint1Pos[0], objectPos[1] - joint1Pos[1])
      theta1 = np.arctan2(objectPos[0] - joint1Pos[0], joint1Pos[1] - objectPos[1])
      # theta2 = 1 - np.arctan2(objectPos[0] - joint2Pos[0], objectPos[1] - joint2Pos[1])
      theta2 = np.pi - np.arctan2(objectPos[0] - joint2Pos[0], joint2Pos[1] - objectPos[1])
      theta3 = np.pi - theta1 - theta2
      distJoint1ToObject = (2.5 * np.sin(theta2))/np.sin(theta3)
    elif objectPos[0] < joint2Pos[0] and objectPos[0] < joint1Pos[0] and objectPos[1] < joint1Pos[1] and objectPos[1] < joint2Pos[1]:
      # theta1 = np.abs(np.arctan2(objectPos[0] - joint1Pos[0], objectPos[1] - joint1Pos[1]))
      theta1 = np.abs(np.arctan2(objectPos[0] - joint1Pos[0], joint1Pos[1] - objectPos[1]))
      # theta2 = 1 - np.abs(np.arctan2(objectPos[0] - joint2Pos[0], objectPos[1] - joint2Pos[1]))
      theta2 = np.pi - np.abs(np.arctan2(objectPos[0] - joint2Pos[0], joint2Pos[1] - objectPos[1]))
      theta3 = np.pi - theta1 - theta2
      distJoint1ToObject = (2.5 * np.sin(theta2))/np.sin(theta3)
    else:
      distJoint1ToObject=-1

    # calculate z and y lengths if distance of object is known:
    if objectPos[0] > joint1Pos[0] and distJoint1ToObject != -1:
      z = np.sin(((np.pi/2) - np.arctan2(objectPos[0] - joint1Pos[0], joint1Pos[1] - objectPos[1]))) * distJoint1ToObject
      y = np.cos(((np.pi/2) - np.arctan2(objectPos[0] - joint1Pos[0], joint1Pos[1] - objectPos[1]))) * distJoint1ToObject
    elif objectPos[0] < joint1Pos[0] and distJoint1ToObject != 1:
      z = np.sin((np.pi/2) - np.abs(np.arctan2(objectPos[0] - joint1Pos[0], joint1Pos[1] - objectPos[1]))) * distJoint1ToObject
      y = np.cos((np.pi/2) - np.abs(np.arctan2(objectPos[0] - joint1Pos[0], joint1Pos[1] - objectPos[1]))) * distJoint1ToObject * -1
    # return (0,0) if object cannot be located in image
    else:
      z = 0.0
      y = 0.0
    return distJoint1ToObject, z, y
 

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
   

  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):

    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)

    # get joint positions
    try:
      joint2Pos = self.detect_blue(self.cv_image1)
      self.cacheBlueCirclePos(joint2Pos)
    except:
      joint2Pos = self.blueCircleCache[-1]
    joint3Pos = joint2Pos
    try:
      joint4Pos =  self.detect_green(self.cv_image1)
      self.cacheGreenCirclePos(joint4Pos)
    except:
      joint4Pos = self.greenCircleCache[-1]
    try:
      endEffectorPos = self.detect_red(self.cv_image1)
      self.cacheRedCirclePos(endEffectorPos)
    except:
      endEffectorPos = self.redCircleCache[-1]

    # get angles
    theta2 = np.arctan2(joint3Pos[0]- joint4Pos[0], joint3Pos[1] - joint4Pos[1])
    theta4 = np.arctan2(joint4Pos[0]- endEffectorPos[0], joint4Pos[1] - endEffectorPos[1]) - theta2

    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
    except CvBridgeError as e:
      print(e)

    # comment out sinusoids for section 3.1
    # adjust joint angles using sinusoidal signals
    # self.joint2=Float64()
    # inputAngle2 = (np.pi/2) * np.sin((np.pi/15) * rospy.get_time())
    # self.joint2.data = inputAngle2
    # self.joint3=Float64()
    # inputAngle3 = (np.pi/2) * np.sin((np.pi/18) * rospy.get_time())
    # self.joint3.data = inputAngle3    
    # # get joint angle 3 from topic if being driven from image2.py
    # # inputAngle3 = self.jointAngle3Data 

    # self.joint4=Float64()
    # inputAngle4 = (np.pi/2) * np.sin((np.pi/20) * rospy.get_time())
    # self.joint4.data = inputAngle4

    # # Publish the results
    # try:
    #   self.robot_joint2_pub.publish(self.joint2)
    #   # self.robot_joint3_pub.publish(self.joint3)
    #   self.robot_joint4_pub.publish(self.joint4)
    # except CvBridgeError as e:
    #   print(e)

    # # publish actual and detected joint angles
    # self.package = Float64()
    # self.package.data = theta2
    # self.jointAngle2.publish(self.package)
    # self.package = Float64()
    # self.package.data = theta4
    # self.jointAngle4.publish(self.package)
    # self.package = Float64()
    # self.package.data = inputAngle2
    # self.actualJointAngle2.publish(self.package)    
    # self.package = Float64()
    # self.package.data = inputAngle4
    # self.actualJointAngle4.publish(self.package)      

    # print joint angles
    # print("Joint Angle 2 Input: {}, Detected Angle: {}".format(inputAngle2, theta2))
    # print("Joint Angle 4 Input: {}, Detected Angle: {}".format(inputAngle4, theta4))

    # get position of circular object
    try:
      objectPos = self.get_object_coordinates(self.cv_image1)
    except:
      objectPos = self.objectCache[-1]
    # position of first joint
    try:
      joint1Pos = self.detect_yellow(self.cv_image1)
      self.cacheYellowCirclePos(joint1Pos)
    except:
      joint1Pos = self.yellowCircleCache[-1]

    # caculate object distance and get z/y coordinates in meters:
    dist, z, y = self.get_distance_base_to_object(joint1Pos, joint2Pos, objectPos)
    # print(z,y)

    # publish estimated position of target
    self.package = Float64()
    self.package.data = z
    self.targetZPosEst.publish(self.package)
    self.package = Float64()
    self.package.data = y
    self.targetYPosEst.publish(self.package)

    endEffectorPosInBase = self.transform(0, 0, 0, np.pi/2) @ \
      self.transform(-np.pi/2, 0, -2.5, -np.pi/2) @ \
      self.transform(0, 0, 0, np.pi/2) @ \
      self.transform(0, 0, -3.5, -np.pi/2) @ \
      self.transform(0, 0, -3, 0)

    print(endEffectorPosInBase)


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


