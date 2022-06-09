
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import pytesseract
import imutils
import time
import cv2
from elements.yolo import OBJ_DETECTION
from runit import VESC
from time import time
from adafruit_servokit import ServoKit

k = ServoKit(channels=16)

k.continuous_servo[0].throttle = 0.6
Object_classes = ['banana']

Object_colors = list(np.random.rand(80,3)*255)
Object_detector = OBJ_DETECTION('weights/yolov5l.pt', Object_classes)

# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
#print(gstreamer_pipeline(flip_method=

# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
cap = cv2.VideoCapture(0)

text = 'banana'
vesc = VESC('/dev/ttyACM0')

if cap.isOpened():
	window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
	# Window
	
	#time.sleep(1)
	while True:
		label = ' '
		ret, frame = cap.read()
		#frame = frame[:380,200:440]
		#cv2.resizeWindow(frame, 640, 640)
		# detection process
		objs = Object_detector.detect(frame)
		#cv2.imshow("CSI Camera", frame)
		# plotting
		if  label != text:
			vesc.run(0.7, 0.1)
		'''elif label == text:
			print(2)
			vesc.run(0.5, 0.2)
			import time
			time.sleep(3)'''
		for obj in objs:
			# print(obj)
			label = obj['label']
			score = obj['score']
			[(xmin,ymin),(xmax,ymax)] = obj['bbox']
			color = Object_colors[Object_classes.index(label)]
			frame = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2) 
			frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX , 0.75, color, 1, cv2.LINE_AA)
			cntr = [(xmin+xmax)/2,(ymin+ymax)/2]
			print(label, cntr, text)
		try:
			if cntr[0] - 320 > 45:
				vesc.run(0.8, 0.1)
			elif cntr[0] - 320 < -45:
				vesc.run(0.2, 0.1)
			else:
				vesc.run(0.5, 0.1)
		except:
			pass
		try:
			area = ((xmax - xmin)*(ymax-ymin))
			print(area)
			if text == 'apple':
				if area >= 26500 and area <= 32000:
					vesc.run(0.5, 0.0)
					k.continuous_servo[0].throttle = -0.15
					break
			elif text == 'banana':
				if area >= 77000 and area <= 100000:
					vesc.run(0.5, 0.0)
					k.continuous_servo[0].throttle = -0.15
					break
		except:
			pass 
		cv2.imshow("CSI Camera", frame)

		#print(area)
		#time.sleep(1)
		keyCode = cv2.waitKey(30)
		if keyCode == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
	while True:
		vesc.run(0.9, -0.2)
else:
	print("Unable to open camera")
