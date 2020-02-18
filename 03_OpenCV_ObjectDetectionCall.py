import os
import time
import cv2
import numpy as np
import requests
import libs

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

yolov3_weights_file = "yolov3.weights"
yolov3_cfg_file = "yolov3.cfg"
yolov3_classes_file = "coco.names"

yolov3_weights_fullfile = os.path.join(asset_path, yolov3_weights_file)
yolov3_cfg_fullfile = os.path.join(asset_path, "darknet", "cfg", yolov3_cfg_file)
yolov3_classes_fullfile = os.path.join(asset_path, "darknet", "data", yolov3_classes_file)

if not os.path.exists(yolov3_weights_fullfile):
  yolov3_classes_url = "https://pjreddie.com/media/files/yolov3.weights"
  with requests.get(yolov3_classes_url, stream=True) as r:
    r.raise_for_status()
    with open(yolov3_weights_fullfile, 'wb') as f:
      for chunk in r.iter_content(chunk_size=8192): 
        if chunk: # filter out keep-alive new chunks
          f.write(chunk)
          # f.flush()

classes = []
with open(yolov3_classes_fullfile, "r") as f:
  classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(yolov3_weights_fullfile, yolov3_cfg_fullfile)

img = cv2.imread(input_fullfile, -1)

img = libs.ObjectDetection(img, net, classes)

cv2.imshow(input_file, img)
key = cv2.waitKey(5000) & 0xFF
if key == ord('s'):
    cv2.imwrite(output_fullfile, img)
cv2.destroyAllWindows()

print("Completed")