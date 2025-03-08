import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('WARNING: Model path is invalid or model was not found. Using default yolo11s.pt model instead.')
    model_path = 'yolo11s.pt'

# Load the model into memory and get label map
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video', 'usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)

    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [file for file in glob.glob(img_source + '/*') if os.path.splitext(file)[1] in img_ext_list]
elif source_type in ['video', 'usb']:
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)

    # Set camera or video resolution if specified by user
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

# Set bounding box colors
bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106), 
               (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Begin inference loop
while True:
    t_start = time.perf_counter()

    # Load frame from image source
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret:
            print('End of video or camera feed. Exiting program.')
            break

    # Resize frame if required
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run inference
    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0  # Initialize object count

    for i in range(len(detections)):
        # Get bounding box coordinates
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get confidence score
        conf = detections[i].conf.item()

        # Draw bounding box if confidence is above threshold
        if conf > float(min_thresh):
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), 
                          (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            object_count += 1  # Increment object count

    # Display FPS (for video or camera)
    if source_type in ['video', 'usb']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    
    # Display object count and image
    cv2.putText(frame, f'Objects detected: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    cv2.imshow('YOLO Detection Results', frame)

    if record:
        recorder.write(frame)

    key = cv2.waitKey(5)

    if key == ord('s') or key == ord('S'):  # Pause
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):  # Save screenshot
        cv2.imwrite('capture.png', frame)

    # Calculate FPS
    t_stop = time.perf_counter()
    frame_rate_calc = 1 / (t_stop - t_start)

    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)

    avg_frame_rate = np.mean(frame_rate_buffer)

# Cleanup
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()
