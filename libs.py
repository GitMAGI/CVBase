import subprocess
import re
import cv2
import numpy as np

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
    regex = re.compile(soi_pattern)
    start_indexes = [m.start(0) for m in re.finditer(soi_pattern, input_data)]

    #print(start_indexes)
    #print(len(start_indexes))

    eoi_pattern = br'\xFF\xD9'
    regex = re.compile(eoi_pattern)
    end_indexes = [m.end(0) for m in re.finditer(eoi_pattern, input_data)]

    for i in range(0, len(start_indexes), 1):
        start = start_indexes[i]
        end = end_indexes[i]
        frame_data = input_data[start:end]
        frames.append(frame_data)
    
    return frames

def ObjectDetection(img, net, classes):
    
    # Detection
    img = cv2.resize(img, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape

    layer_names = net.getLayerNames()
    #print(layer_names)
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #print(output_layers)

    blob_scale_factor = 0.00392
    blob_dim = (416, 416)

    blob = cv2.dnn.blobFromImage(img, blob_scale_factor, blob_dim, (0, 0, 0), True, crop=False)

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
            if confidence > 0.5:
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
    
    return img