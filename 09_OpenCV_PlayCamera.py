import numpy as np
import cv2
import os
import imutils
import time
import libs

print("Starting ...")
print("OpenCV Version:", cv2.__version__)
print("NumPy Version:", np.__version__)

print("OpenCV VideoCapture starting ...")
cap = cv2.VideoCapture(0)
print("OpenCV VideoCapture started!")

fps = cap.get(cv2.CAP_PROP_FPS)
print("OpenCV VideoCapture FPS:", fps)

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret: #If ret is None means we reached the end of the stream!
        break
    
    elapsed_time = time.time() - start_time
    #print(libs.ElapsedTime2String(elapsed_time))

    # show the output frame
    cv2.imshow("Input Camera", frame)

    waitTime = int(fps) - int(round(elapsed_time * 1000))
    if waitTime < 0: waitTime = 1

    key = cv2.waitKey(waitTime) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("OpenCV VideoCapture released!")

print("Completed!")