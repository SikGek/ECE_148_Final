#It is capable of (1) running at near real-time at 13 FPS on 720p images and (2) obtains state-of-the-art text detection accuracy.
#The EAST pipeline is capable of predicting words and lines of text at arbitrary orientations on 720p images, and furthermore, can run at 13 FPS, according to the authors.

# import the necessary packages
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
import time
from adafruit_servokit import ServoKit

k = ServoKit(channels=16)
if k.continuous_servo[0].throttle != 0.6:
	k.continuous_servo[0].throttle = 0.6

vesc = VESC('/dev/ttyACM0')
#pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ihyun\Desktop\tesseract\tesseract.exe'

def decode_predictions(scores, geometry):

	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):

		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		for x in range(0, numCols):

			if scoresData[x] < args["min_confidence"]:
				continue

			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	return (rects, confidences)


ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, required=True,
	help="path to input EAST text detector")
ap.add_argument("-v", "--video", type=str,
	help="path to optinal input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())



(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

items = ["apple", 'banana', 'orange']


if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

else:
	vs = cv2.VideoCapture(0)

fps = FPS().start()


while True:

	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	if frame is None:
		break

	frame = imutils.resize(frame, width=1000)
	orig = frame.copy()

	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)

	frame = cv2.resize(frame, (newW, newH))

	blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	results = []
	for (startX, startY, endX, endY) in boxes:

		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)


		dX = int((endX - startX) * args["padding"])
		dY = int((endY - startY) * args["padding"])

		startX = max(0, startX - dX)
		startY = max(0, startY - dY)
		endX = min(W, endX + (dX * 2))
		endY = min(H, endY + (dY * 2))

		roi = orig[startY:endY, startX:endX]
		config = ("-l eng --oem 3 --psm 7")
		text = pytesseract.image_to_string(roi, config=config)

		results.append(((startX, startY, endX, endY), text))

		results = sorted(results, key=lambda r:r[0][1])
		for ((startX, startY, endX, endY), text) in results:
			text = text.rstrip()
			print("OCR TEXT")
			print("========")
			print("|" + text + "|")
			if text in items:
				break

	try:
		if text in items:
			break
	except:
		pass
	fps.update()

	cv2.imshow("Text Detection", orig)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break
	if key == ord("z"):
		exit()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

'''if not args.get("video", False):
	vs.stop()

else:
	vs.release()'''

cv2.destroyAllWindows()
#print(text)

'''from donkeycar.parts.actuator import VESC
VESC_SERIAL_PORT = '/dev/ttyACM0'
VESC_MAX_SPEED_PERCENT = 0.2
VESC_HAS_SENSOR = False
VESC_START_HEARTBEAT= True
VESC_BAUDRATE= 115200
VESC_TIMEOUT= 0.05
DONKEY_GYM = False
#logger.info("Creating VESC at port {}".format(cfg.VESC_SERIAL_PORT))
vesc = VESC(cfg.VESC_SERIAL_PORT,
	cfg.VESC_MAX_SPEED_PERCENT,
	cfg.VESC_HAS_SENSOR,
	cfg.VESC_START_HEARTBEAT,
	cfg.VESC_BAUDRATE, 
	cfg.VESC_TIMEOUT
)'''
Object_classes = [text]
#Object_classes = ['person']

Object_colors = list(np.random.rand(80,3)*255)

Object_detector = OBJ_DETECTION('weights/yolov5l.pt', Object_classes)

#cap = cv2.VideoCapture(4)
#window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
# Window

#window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
# Window
label = ' '
serv = ' '
while True:
	frame = vs.read()
	#frame = frame[:380,200:440]
	#cv2.resizeWindow(frame, 640, 640)
	# detection process
	objs = Object_detector.detect(frame)
	#cv2.imshow("CSI Camera", frame)
	# plotting
	if  label != text:
		vesc.run(0.65, 0.1)
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
		if cntr[0] - 320 > 50:
			vesc.run(0.7, 0.1)
		elif cntr[0] - 320 < -50:
			vesc.run(0.3, 0.1)
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
			if area >= 65000 and area <= 100000:
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
#else:

#k.continuous_servo[0].throttle = 1
vs.stop()
cv2.destroyAllWindows()
while True:
    vesc.run(0.9, -0.1)