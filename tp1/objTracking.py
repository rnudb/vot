"""
Create the main file of this project that will be execute to track an object (objTracking.py)
2.1 Import function detect() and KalmanFilter
"""

import cv2
import numpy as np
from KalmanFilter import KalmanFilter
from matplotlib import pyplot as plt
from data.Detector import detect
import time

rect_padding = 10

# Create the object of the class Kalman filter and set parameters values as:
#  dt=0.1, u_x=1, u_y=1, std_acc=1, x_dt_meas=0.1, y_dt_meas=0.1
# You can also try to set other values and observe the performance.
dt = 0.1
u_x = 1
u_y = 1
std_acc = 1
x_dt_meas = 0.1
y_dt_meas = 0.1

kalman = KalmanFilter(dt, u_x, u_y, std_acc, x_dt_meas, y_dt_meas)

# Create video capture object
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('data/randomball.avi')


# Object Detection Integration. Use provided object detection code to detect black cercle in each
# frame.
trajectory = []

while True:
    # Pause for a short time such that we are doing 60fps
    time.sleep(0.02)

    ret, frame = cap.read()
    if not ret:
        break

    # Detect the circle in the frame
    circles = detect(frame)

    # If centroid is detected then track it.
    for circle in circles:
        # Call the Kalman prediction function and the Kalman filter update function
        kalman.predict()

        kalman.update(circle[:2])

        # Draw the circle on the frame
        x, y, radius = map(int, circle)
        cv2.circle(frame, (x, y), radius, (0, 255, 0), 1)

        # Draw a blue rectangle as the detected object position
        x, y = map(int, kalman.predict()[:2])
        cv2.rectangle(
            frame,
            (x - rect_padding, y - rect_padding),
            (x + rect_padding, y + rect_padding),
            (255, 0, 0),
            2,
        )

        # Draw a red rectangle as the estimated object position
        x, y = map(int, kalman.update(circle[:2])[:2])
        cv2.rectangle(
            frame,
            (x - rect_padding, y - rect_padding),
            (x + rect_padding, y + rect_padding),
            (0, 0, 255),
            2,
        )

        # Draw trajectory
        trajectory.append((x, y))
        if len(trajectory) > 50:
            trajectory.pop(0)
        for i in range(len(trajectory) - 1):
            cv2.line(frame, trajectory[i], trajectory[i + 1], (255, 255, 0), 2)

    # Show the frame
    frame = cv2.resize(frame, (1080, 720))
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
