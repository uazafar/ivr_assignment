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
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()

    # initialize a publisher to send joints' angular position to the robot
    self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10) 
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)    


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

  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)

    # get join positions
    a = self.pixel2meter(self.cv_image1)
    joint2Pos = a * self.detect_blue(self.cv_image1)
    joint3Pos = joint2Pos
    joint4Pos = a * self.detect_green(self.cv_image1)
    endEffectorPos = self.pixel2meterEndEff(self.cv_image1) * self.detect_red(self.cv_image1)

    # get angles
    theta2 = np.arctan2(joint3Pos[0]- joint4Pos[0], joint3Pos[1] - joint4Pos[1])
    theta4 = np.arctan2(joint4Pos[0]- endEffectorPos[0], joint4Pos[1] - endEffectorPos[1])



    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
    except CvBridgeError as e:
      print(e)

    # adjust joint angles using sinusoidal signals
    self.joint2=Float64()
    inputAngle2 = (np.pi/2) * np.sin((np.pi/15) * rospy.get_time())
    self.joint2.data = inputAngle2
    self.joint4=Float64()
    inputAngle4 = (np.pi/2) * np.sin((np.pi/20) * rospy.get_time())
    self.joint4.data = inputAngle4

    # Publish the results
    try:
      self.robot_joint2_pub.publish(self.joint2)
      self.robot_joint4_pub.publish(self.joint4)
    except CvBridgeError as e:
      print(e)


    # # print joint angles
    # print("Joint Angle 2 Input: {}, Detected Angle: {}".format(inputAngle2, theta2))
    # print("Joint Angle 4 Input: {}, Detected Angle: {}".format(inputAngle4, theta4))

    # detect object
    # gray = cv2.cvtColor(self.cv_image1, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray, 5)

    # Threshold the HSV image to get only blue colors
    im = cv2.inRange(self.cv_image1, (50,50,100), (255,255,255))

    res = cv2.bitwise_and(self.cv_image1, self.cv_image1, mask= im)

    # rows = gray.shape[0]
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
    #                            param1=100, param2=30,
    #                            minRadius=1, maxRadius=30)
    
    
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         center = (i[0], i[1])
    #         # circle center
    #         cv2.circle(self.cv_image1, center, 1, (0, 100, 100), 3)
    #         # circle outline
    #         radius = i[2]
    #         cv2.circle(self.cv_image1, center, radius, (255, 0, 255), 3)


    im1=cv2.imshow('window1', res)
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


