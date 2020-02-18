import os
import time
import libs

print("Starting ...")

input_path = "input"
output_path = "output"

input_file = "20190827_215900.mp4"
input_filename, input_fileextension = os.path.splitext(input_file)
input_fullfile = os.path.join(input_path, input_file)

input_w = int(1920/5)
input_h = int(1080/5)
frames = libs.FFMPEGExtractJPGFromMP4(input_fullfile, input_w, input_h)
print("Extracted %d jpg frames" % len(frames))



print("Completed")