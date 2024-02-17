import cv2
import pathlib
import time
import argparse
import os

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='path to input video',
    default='input/video_1.mp4'
)
parser.add_argument(
    '-t', '--template', help='path to the template',
    default='input/video_1_template.jpg'
)
args = vars(parser.parse_args())

# Read the video input.
cap = cv2.VideoCapture(args['input'])

if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# Get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# String name with which to save the resulting video.
save_name = str(pathlib.Path(
    args['input']
)).split(os.path.sep)[-1].split('.')[0]
# define codec and create VideoWriter object 
out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))

# Read the template in grayscale format.
template = cv2.imread(args['template'], 0)
w, h = template.shape[::-1]

frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.

# Read until end of video.
while(cap.isOpened()):
    # Capture each frame of the video.
    ret, frame = cap.read()
    if ret:
        image = frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        start_time = time.time()
        # Apply template Matching.
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        end_time = time.time()
        
        # Get the current fps.
        if end_time != start_time:
            print(end_time - start_time)
            fps = 1 / (end_time - start_time)
        else:
            fps = 0
        # Add `fps` to `total_fps`.
        total_fps += fps
        # Increment frame count.
        frame_count += 1
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # Top left x and y coordinates.
        x1, y1 = max_loc
        # Bottom right x and y coordinates.
        x2, y2 = (x1 + w, y1 + h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 
                    2, lineType=cv2.LINE_AA)
        cv2.imshow('Result', frame)
        out.write(frame)
        # Press `q` to exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release VideoCapture() object.
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()