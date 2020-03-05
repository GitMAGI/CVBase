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