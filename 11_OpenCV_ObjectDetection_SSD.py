import numpy as np
import cv2
import os
import imutils
import time
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

# LOAD the Net -- Begin
blob_scale_factor = 0.007843
blob_dim = (416, 416)
blob_mean = 127.5
confidence_limit = .5
prototxt_file = "MobileNetSSD_deploy.prototxt.txt"
caffe_model_file = "MobileNetSSD_deploy.caffemodel"
prototxt_fullfile = os.path.join(asset_path, prototxt_file)
caffe_model_fullfile = os.path.join(asset_path, caffe_model_file)
classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNetFromCaffe(prototxt_fullfile, caffe_model_fullfile)
# LOAD the Net -- End

counter = 0
fps_start_time = time.time()
while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret: #If ret is None means we reached the end of the stream!
        break
    counter += 1
    
    frame = oDs.SSD_MobileNet(frame, blob_dim, blob_scale_factor, blob_mean, net, classes, colors, confidence_limit, False)

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
fps_end_time = time.time()
print("OpenCV VideoCapture Avg FPS: %.1f" % (counter / (fps_end_time - fps_start_time)))

cap.release()
cv2.destroyAllWindows()
print("OpenCV VideoCapture released!")

print("Completed!")