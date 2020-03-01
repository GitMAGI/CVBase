import subprocess
import re
import time
import cv2
import numpy as np
import os
import requests
import math
import imutils

def FFMPEGExtractJPGFromMP4(input_fullfile, input_w, input_h):
    # Command for extract a sequence of jpgs from a video file
    # ffmpeg -i .\input\video.mp4 -c:v mjpeg -f image2pipe -s Width x Height pipe:1
    cmd = [
        'ffmpeg', '-i', input_fullfile, '-c:v', 'mjpeg', '-q:v', '1', '-f', 'image2pipe', '-s', '{}x{}'.format(input_w, input_h), 'pipe:1'
    ]
    input_process = subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE)
    [input_data, input_err] = input_process.communicate(input = input_process)
    if input_err:
        print(input_err)
        return []
    print("Got %d Bytes of data" % len(input_data))

    # Gather an array of JPGs
    # A JPG is delimited by 2 Sequences:
    #  SOI (Start of Image) 0xFF 0xD8
    #  EOI (End of Image)   0xFF 0xD9
    frames = []
    soi_pattern = br'\xFF\xD8'
    #regex = re.compile(soi_pattern)
    start_indexes = [m.start(0) for m in re.finditer(soi_pattern, input_data)]

    #print(start_indexes)
    #print(len(start_indexes))

    eoi_pattern = br'\xFF\xD9'
    #regex = re.compile(eoi_pattern)
    end_indexes = [m.end(0) for m in re.finditer(eoi_pattern, input_data)]

    for i in range(0, len(start_indexes), 1):
        start = start_indexes[i]
        end = end_indexes[i]
        frame_data = input_data[start:end]
        frames.append(frame_data)
    
    return frames

def ObjectDetection(img, blob_size, blob_scale_factor, net, classes, confidence_limit = .5):
    start_time = time.time()

    # Detection
    img = cv2.resize(img, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape

    layer_names = net.getLayerNames()
    #print(layer_names)
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #print(output_layers)

    blob = cv2.dnn.blobFromImage(img, blob_scale_factor, blob_size, (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_limit:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]]) + " ({0:.2f})".format(confidences[i])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            cv2.rectangle(img, (x,y-10), (x + w, y), color, cv2.FILLED)
            cv2.putText(img, label, (x, y - 1), font, 0.6, (255,255,255), 1)
    
    elapsed_time = time.time() - start_time
    print(ElapsedTime2String(elapsed_time))

    return img

def ObjectDetectionDev(img, blob_size, blob_scale_factor, net, classes, confidence_limit = .5):
    start_time = time.time()

    # Detection
    img = cv2.resize(img, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape

    layer_names = net.getLayerNames()
    #print(layer_names)
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #print(output_layers)

    blob = cv2.dnn.blobFromImage(img, blob_scale_factor, blob_size, (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    font = cv2.FONT_HERSHEY_PLAIN

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_limit:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                label = str(classes[class_id] + " ({0:.2f})".format(confidence))
                color = colors[class_id]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                cv2.rectangle(img, (x,y-10), (x + w, y), color, cv2.FILLED)
                cv2.putText(img, label, (x, y - 1), font, 0.6, (255,255,255), 1)
    
    elapsed_time = time.time() - start_time
    print(ElapsedTime2String(elapsed_time))

    return img

def ObjectDetectionAdvanced(img, blob_size, blob_scaling_factor, blob_mean, net, classes, confidence_limit = .5):
    start_time = time.time()

    max_width = math.ceil(blob_size[1]/100)*100
    img = imutils.resize(img, width=max_width)

    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, blob_size), blob_scaling_factor, blob_size, blob_mean)

    net.setInput(blob)
    detections = net.forward()

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > confidence_limit:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
            cv2.rectangle(img, (startX, startY), (endX, endY), colors[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
    
    elapsed_time = time.time() - start_time
    print(ElapsedTime2String(elapsed_time))

    return img

def ElapsedTime2String(elapsed_time):
    millis = int(round(elapsed_time * 1000))
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + str(".%03d" % millis)

def DownloadIfNotExists(full_pathfile, url_file):
    if not os.path.exists(full_pathfile):
        with requests.get(url_file, stream=True) as r:
            r.raise_for_status()
            with open(full_pathfile, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        # f.flush()