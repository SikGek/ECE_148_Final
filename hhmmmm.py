import cv2
import pytesseract
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import argparse

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ihyun\Desktop\tesseract\tesseract.exe'

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

net = cv2.dnn.readNet(args["east"])
vs = VideoStream(src=0).start()
time.sleep(1.0)
(newW, newH) = (args["width"], args["height"])
while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    frame = cv2.resize(frame, (newW, newH))
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    orig = frame.copy()
    config = ("-l eng --oem 3 --psm 7")
    blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    text = pytesseract.image_to_string(frame, config=config)
    print(text)

    cv2.imshow("Text Detection", orig)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
	    break

vs.release()
cv2.destroyAllWindows()
