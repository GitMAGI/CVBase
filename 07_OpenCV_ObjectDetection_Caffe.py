import os
import time
import cv2
import numpy as np
import requests
from matplotlib import pyplot as plt
import libs

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

input_w = int(1920/4)
input_h = int(1080/4)
frames = libs.FFMPEGExtractJPGFromMP4(input_fullfile, input_w, input_h)
print("Extracted {} frames".format(len(frames)))

blob_scale_factor = 0.007843
blob_dim = (input_w, input_h)
blob_mean = 127.5
confidence_limit = .5

prototxt_file = "MobileNetSSD_deploy.prototxt.txt"
caffe_model_file = "MobileNetSSD_deploy.caffemodel"

prototxt_fullfile = os.path.join(asset_path, prototxt_file)
caffe_model_fullfile = os.path.join(asset_path, caffe_model_file)

classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

net = cv2.dnn.readNetFromCaffe(prototxt_fullfile, caffe_model_fullfile)

for frame in frames:
    img = cv2.imdecode(np.frombuffer(frame, dtype = np.uint8), -1)    
    img = libs.ObjectDetectionAdvanced(img, blob_dim, blob_scale_factor, blob_mean, net, classes, confidence_limit)
    cv2.imshow(input_file, img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

print("Completed")