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


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
  

    # define a cache to store positions of circles
    self.greenCircleCache = []
    self.redCircleCache = []
    self.blueCircleCache = []
    self.yellowCircleCache = []
    # if object not visible straight away, initialise with the following position (behind robot)
    self.objectCache = [[400,400]]
    self.orangeSquareCache = [[500,354]]

    # calculate the meters per pixel at first iteration
    self.meterPerPixel = None

    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera2 to a topic named image_topic2
    self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()

    # initialize a publisher to send joints' angular position to the robot
    self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10) 
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)   

    # set up publisher   
    self.jointAngle3 = rospy.Publisher("jointAngle3", Float64, queue_size=10)
    self.targetYPosEst = rospy.Publisher("targetYPosEst", Float64, queue_size=10)
    self.orangeYPosEst = rospy.Publisher("orangeYPosEst", Float64, queue_size=10)

    # position of end effector as calculated using cv
    self.endEffYPos = rospy.Publisher("endEffYPos", Float64, queue_size=10)

    self.time = rospy.get_time()


  # functions to store historic positions of circles (max. 1000 positions)

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

  # function to get coordinates of orange sphere
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
      return keypoints[0].pt
    else:
      # return last known position of object
      return self.objectCache[-1]

  # function to measure joint angle 3
  def getTheta3(self, image):
    # get joint positions
    try:
      joint2Pos = self.detect_blue(image)
      self.cacheBlueCirclePos(joint2Pos)
    except:
      joint2Pos = self.blueCircleCache[-1]
    try:
      joint4Pos =  self.detect_green(image)
      self.cacheGreenCirclePos(joint4Pos)
    except:
      joint4Pos = self.greenCircleCache[-1]

    # get angle
    theta3 = np.arctan2(joint2Pos[0]- joint4Pos[0], joint2Pos[1] - joint4Pos[1])*-1  
    
    return theta3  



  # function to get coordinates of orange cube
  def get_orange_square_coordinates(self, image):
    # Threshold the HSV image to get only orange colors (of object)
    mask = cv2.inRange(image, (0,20,100), (40,100,150))
    res = cv2.bitwise_and(image, image, mask= mask)

    # convert image to greyscale
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # create parameters for blob detector to detect circles
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.maxArea = 350
    params.filterByInertia = 1.0
    params.filterByConvexity = False
    params.filterByCircularity = 1.0
    params.minCircularity = 0.00
    params.maxCircularity = 0.87

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

  # function to get coordinates of orange cube (in meters)
  def getOrangeSquareCoordinates(self, image):
    try:
      orangeSquare = self.get_orange_square_coordinates(image)
      self.cacheOrangeSquarePos(orangeSquare)
    except:
      orangeSquare = self.objectCache[-1]
    # position of first joint
    try:
      joint1Pos = self.detect_yellow(image)
      self.cacheYellowCirclePos(joint1Pos)
    except:
      joint1Pos = self.yellowCircleCache[-1]


    # calculate distance from base to object
    distBaseToObjectPixels = np.sum((joint1Pos - orangeSquare)**2)
    distBaseToObjectMeters = self.meterPerPixel * np.sqrt(distBaseToObjectPixels)    
    baseToTargetAngle = np.arctan2(joint1Pos[0]- orangeSquare[0], joint1Pos[1] - orangeSquare[1])
    orangeSquareZ = distBaseToObjectMeters*np.cos(baseToTargetAngle)
    orangeSquareY = distBaseToObjectMeters*np.sin(baseToTargetAngle)    
    
    return orangeSquareY, orangeSquareZ   


  # function to get coordinates of end effector (in meters)
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
    distBaseToEEPixels = np.sum((joint1Pos - endEffPos)**2)   
    distBaseToEEMeters = self.meterPerPixel * np.sqrt(distBaseToEEPixels) 
    baseToEEAngle = np.arctan2(joint1Pos[0]- endEffPos[0], joint1Pos[1] - endEffPos[1])
    endEffZ = distBaseToEEMeters*np.cos(baseToEEAngle)
    endEffY = distBaseToEEMeters*np.sin(baseToEEAngle)    
    
    return endEffY, endEffZ 

  # function to get coordinates of orange sphere (in meters)
  def getObjectCoordinates(self, image):
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
    distBaseToObjectPixels = np.sum((joint1Pos - objectPos)**2)
    distBaseToObjectMeters = self.meterPerPixel * np.sqrt(distBaseToObjectPixels)    
    baseToTargetAngle = np.arctan2(joint1Pos[0]- objectPos[0], joint1Pos[1] - objectPos[1])
    targetZ = distBaseToObjectMeters*np.cos(baseToTargetAngle)
    targetY = distBaseToObjectMeters*np.sin(baseToTargetAngle)    
    
    return targetY, targetZ   


  # Recieve data, process it, and publish
  def callback2(self,data):
    # Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # Publish the results
    try: 
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
    except CvBridgeError as e:
      print(e)

    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)


    # calculate metes per pixel and store value
    if not self.meterPerPixel:
      self.meterPerPixel = self.pixel2meterYellowToRed(self.cv_image2)

    # SECTION 2.1
    theta3 = self.getTheta3(self.cv_image2)
 

    #SECTION 2.2:
    # get position of circular object
    targetY, _ = self.getObjectCoordinates(self.cv_image2)

    orangeSquareY, _ = self.getOrangeSquareCoordinates(self.cv_image2)

    # publish estimated position of orange cube
    self.package = Float64()
    self.package.data = orangeSquareY
    self.orangeYPosEst.publish(self.package)

    # publish estimated position of target
    self.package = Float64()
    self.package.data = targetY
    self.targetYPosEst.publish(self.package)

    # publish detected joint angles
    self.package = Float64()
    self.package.data = theta3
    self.jointAngle3.publish(self.package)  



    # SECTION 3.2

    # publish position of end effector as calculated using cv
    endEffY, _  = self.getEndEffectorCoordinates(self.cv_image2)
    self.package = Float64()
    self.package.data = endEffY
    self.endEffYPos.publish(self.package)




    # im2=cv2.imshow('window2', self.cv_image2)
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


