import time
import cv2
import numpy as np
import os
import requests
import math
import imutils
import libs

# SSD (Single Shot MultiBox Detector)
# It's an implementation of a SSD with a MobileNet for the Classification
def SSD_MobileNet(img, blob_size, blob_scaling_factor, blob_mean, net, classes, colors, confidence_limit = .5, logger = False):
    start_time = time.time()

    max_width = math.ceil(blob_size[1]/100)*100
    img = imutils.resize(img, width=max_width)

    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, blob_size), blob_scaling_factor, blob_size, blob_mean)

    net.setInput(blob)
    detections = net.forward()

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
    if logger:
        print("Frame Computing time:", libs.ElapsedTime2String(elapsed_time))

    return img

# YOLO v3 (You only look once)
# You use a generic Classifier???
def YOLOv3_Generic(img, blob_size, blob_scale_factor, net, layer_names, classes, colors, confidence_limit = .5, logger = False):    
    start_time = time.time()

    (H, W) = img.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(img, blob_scale_factor, blob_size, swapRB=True, crop=False)
    net.setInput(blob)

    start_net_computing = time.time()
    
    layerOutputs = net.forward(layer_names)
    net_computing_elapsed_time = time.time() - start_net_computing
    if logger:
        print("Net Computing time:", libs.ElapsedTime2String(net_computing_elapsed_time))

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_limit:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    score_treshold = 0.5
    nms_treshold = 0.4
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_treshold, nms_treshold)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    elapsed_time = time.time() - start_time
    if logger:
        print("Frame Computing time:", libs.ElapsedTime2String(elapsed_time))

    return img