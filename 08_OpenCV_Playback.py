import numpy as np
import cv2
import os
import imutils
import time
import libs

print("Starting ...")
print("OpenCV Version:", cv2.__version__)
print("NumPy Version:", np.__version__)

input_path = "input"
output_path = "output"
if not os.path.exists(output_path):
    os.makedirs(output_path)
asset_path = "asset"

input_file = "traffic02.mp4"
input_filename, input_fileextension = os.path.splitext(input_file)
output_file = input_filename + "_Detection_" + time.strftime("%Y%m%d-%H%M%S") + input_fileextension

input_fullfile = os.path.join(input_path, input_file)
output_fullfile = os.path.join(output_path, output_file)

print("Input file:", input_fullfile)
print("Output file:", output_fullfile)

print("OpenCV VideoCapture starting ...")
cap = cv2.VideoCapture(input_fullfile)
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
    cv2.imshow(input_file, frame)

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