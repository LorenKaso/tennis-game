# table_tennis_analyzer.py

from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO('yolov8n-pose.pt') 

# Set start and end frame constants
START_FRAME = 0
END_FRAME = 200


# Load video and get properties
cap = cv2.VideoCapture('input.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_with_detections.mp4', fourcc, fps, (width, height))

if START_FRAME > 0:
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

frame_num = START_FRAME

# Process frames from the video
while cap.isOpened():
    print(f"Frame number: {frame_num}")
    frame_num += 1
    if frame_num > END_FRAME:
        break
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break

    if END_FRAME != 0 and frame_num > END_FRAME:
        print(f"Reached END_FRAME: {END_FRAME}")
        break
    
    # Resize frame for faster processing
    scale = 0.55
    new_width = int(frame.shape[1] * scale)
    new_height = int(frame.shape[0] * scale)

    resized_frame = cv2.resize(frame, (new_width, new_height))
    results = model(resized_frame, imgsz=1500, conf=0.25)

    # Annotate and write frame to output
    annotated_frame = results[0].plot()
    annotated_frame_resized = cv2.resize(annotated_frame, (width, height))
    out.write(annotated_frame_resized)
    frame_num += 1

# Release video resources
cap.release()
out.release()
cv2.destroyAllWindows()
