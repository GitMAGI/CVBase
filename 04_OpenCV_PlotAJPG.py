import os
import time
import cv2
import numpy as np
import requests
from matplotlib import pyplot as plt
from PIL import Image

print("Starting ...")
print("OpenCV Version:", cv2.__version__)
print("NumPy Version:", np.__version__)

input_path = "input"
output_path = "output"
if not os.path.exists(output_path):
    os.makedirs(output_path)
asset_path = "asset"

input_file = "traffic01.jpg"
input_filename, input_fileextension = os.path.splitext(input_file)
output_file = input_filename + "_Detection_" + time.strftime("%Y%m%d-%H%M%S") + input_fileextension

input_fullfile = os.path.join(input_path, input_file)
output_fullfile = os.path.join(output_path, output_file)

print("Input file:", input_fullfile)
print("Output file:", output_fullfile)

with open(input_fullfile, 'rb') as f:
    frame = f.read()

img = cv2.imdecode(np.fromstring(frame, dtype = np.uint8), -1)                 
cv2.imshow(input_file, img)  
cv2.waitKey(1500)
cv2.destroyAllWindows()

print("Completed")