from adafruit_servokit import ServoKit

k = ServoKit(channels=16)

k.continuous_servo[0].throttle = -0.1

