import os
import time
import cv2
import numpy as np
import requests
from matplotlib import pyplot as plt
import libs
import objectDetections as oDs

print("Starting ...")
print("OpenCV Version:", cv2.__version__)
print("NumPy Version:", np.__version__)

input_path = "input"
output_path = "output"
if not os.path.exists(output_path):
    os.makedirs(output_path)
asset_path = "asset"

#input_file = "skidrow01.mp4"
#input_file = "traffic01.mp4"
#input_file = "video.mp4"
input_file = "traffic02.mp4"
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

yolov3_weights_url = "https://pjreddie.com/media/files/yolov3.weights"
libs.DownloadIfNotExists(yolov3_weights_fullfile, yolov3_weights_url)

classes = []
with open(yolov3_classes_fullfile, "r") as f:
  classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#net = cv2.dnn.readNet(yolov3_weights_fullfile, yolov3_cfg_fullfile)
net = cv2.dnn.readNetFromDarknet(yolov3_cfg_fullfile, yolov3_weights_fullfile)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

input_w = int(1920/5)
input_h = int(1080/5)
frames = libs.FFMPEGExtractJPGFromMP4(input_fullfile, input_w, input_h)

print("Extracted {} frames".format(len(frames)))

blob_scale_factor = 0.00392
blob_dim = (416, 416)
confidence_limit = .5

for frame in frames:
    img = cv2.imdecode(np.frombuffer(frame, dtype = np.uint8), -1)    
    img = oDs.YOLOv3_Generic(img, blob_dim, blob_scale_factor, net, output_layers, classes, colors, confidence_limit, True)
    cv2.imshow(input_file, img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

print("Completed")