## Introduction
Hi guys, we're team 3 from ECE/MAE 148 from Spring 2022, my name is specifically Ihyun Leo Park, and this is our final project.

We implemented a text detection model which takes in text as an input for a class for our YOLO object detection, which then goes and finds the specified object.

## Download Model
Select the desired model based on model size, required speed, and accuracy.
You can find available models [**here**](https://github.com/ultralytics/yolov5/releases) in the **Assets** section.
Download the model using the command below and move it to the **weights** folder.
```
$ cd weights
$ wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l.pt
```

## How to Run the Code
Just run the code "ece_148_final_team3.py --east frozen_east_text_detection.pb" as such and show the car either an apple, banana or orange and you will be able to see it in no time.

## References

https://github.com/amirhosseinh77/JetsonYolo
https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV
