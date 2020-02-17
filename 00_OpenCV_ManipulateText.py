import os
import time
import cv2
import numpy as np

print("Starting ...")
print("OpenCV Version:", cv2.__version__)
print("NumPy Version:", np.__version__)

input_path = "input"
output_path = "output"
if not os.path.exists(output_path):
    os.makedirs(output_path)

input_file = "skidrow01.jpg"
input_filename, input_fileextension = os.path.splitext(input_file)
output_file = input_filename + "_Text_" + time.strftime("%Y%m%d-%H%M%S") + input_fileextension

input_fullfile = os.path.join(input_path, input_file)
output_fullfile = os.path.join(output_path, output_file)

print("Input file:", input_fullfile)
print("Output file:", output_fullfile)

img = cv2.imread(input_fullfile, -1)

img = cv2.rectangle(img, (300,300), (600,600), (0, 0, 255), 2)
img = cv2.circle(img, (400,600), 50, (255,0,0), 2)
img = cv2.putText(img, 'OpenCV', (300,600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4, cv2.LINE_AA)

cv2.imshow(input_file, img)
key = cv2.waitKey(5000) & 0xFF
if key == ord('s'):
    cv2.imwrite(output_fullfile, img)
cv2.destroyAllWindows()

print("Completed")