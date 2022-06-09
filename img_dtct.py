import cv2
import numpy as np
from elements.yolo import OBJ_DETECTION

Object_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush' ]

Object_colors = list(np.random.rand(80,3)*255)
Object_detector = OBJ_DETECTION('weights/yolov5s1.pt', Object_classes)

# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
#print(gstreamer_pipeline(flip_method=0))

frame = cv2.imread('banana.jpg')
# detection process
objs = Object_detector.detect(frame)
window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
frame = cv2.resize(frame, (640,640))
# plotting
for obj in objs:
    # print(obj)
    label = obj['label']
    score = obj['score']
    [(xmin,ymin),(xmax,ymax)] = obj['bbox']
    color = Object_colors[Object_classes.index(label)]
    frame = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2) 
    frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX , 0.75, color, 1, cv2.LINE_AA)
    #print(xmin, xmax, ymin, ymax)
    cv2.imshow("CSI Camera", frame)
    cv2.waitKey(30)
    
cv2.imshow("CSI Camera", frame)
cv2.waitKey(0)

cv2.destroyAllWindows()
