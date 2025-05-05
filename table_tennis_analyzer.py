# table_tennis_analyzer.py
from ultralytics import YOLO
import numpy as np
import cv2
#heat-map
import csv
from ultralytics import YOLO
import heatmap  

# Load YOLO model 
model = YOLO('yolo11n-pose.pt') 

# Set start and end frame constants
START_FRAME = 0
END_FRAME = 1500

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

# Initialize CSV file for player positions
csv_file = open("player_positions.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "player1_position_x", "player1_position_y", "player2_position_x", "player2_position_y"])

#background_frame heatmap
background_frame = None
cap.set(cv2.CAP_PROP_POS_FRAMES, 20) 
ret, background_frame = cap.read()
if ret:
    cv2.imwrite("background_frame.png", background_frame)
cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME) 

player1_ref = None 

def is_valid_frame(boxes, width, height):
    if len(boxes) != 2:
        return False

    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        box_height = y2 - y1
        box_width = x2 - x1

        if box_height < height * 0.2:   
            return False
        if box_width < width * 0.05:   
            return False
        if y1 < height * 0.1:          
            return False
        if box_width / box_height < 0.2:  
            return False

    h0 = boxes[0][3] - boxes[0][1]
    h1 = boxes[1][3] - boxes[1][1]
    if abs(h0 - h1) > height * 0.15:
        return False

    return True


while cap.isOpened():
    print(f"Frame number: {frame_num}")
    if frame_num >= END_FRAME and END_FRAME != 0:
        break

    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break

    # Resize frame
    scale = 0.55
    resized_frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

    # YOLO Detection
    results = model(resized_frame, imgsz=1500, conf=0.25)
    boxes = results[0].boxes.xyxy  # bounding boxes
    boxes = sorted(boxes, key=lambda b: b[2] - b[0], reverse=True)[:2]

    if is_valid_frame(boxes, width, height):
        boxes = sorted(boxes, key=lambda b: b[2] - b[0], reverse=True)[:2]

        player_centers = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            foot_x = (x1 + x2) / 2 / scale
            foot_y = y2 / scale
            player_centers.append((foot_x, foot_y))

        if len(player_centers) == 2:
            if player1_ref is None:
                player_centers.sort(key=lambda p: p[0])  
            player1_ref = player_centers[0]  

            d0 = np.linalg.norm(np.array(player_centers[0]) - np.array(player1_ref))
            d1 = np.linalg.norm(np.array(player_centers[1]) - np.array(player1_ref))
            if d0 < d1:
                p1, p2 = player_centers[0], player_centers[1]
            else:
                p1, p2 = player_centers[1], player_centers[0]

            csv_writer.writerow([frame_num, p1[0], p1[1], p2[0], p2[1]])

        # Annotate and write only valid frames
        annotated_frame = results[0].plot()
        output_frame = cv2.resize(annotated_frame, (width, height))
        out.write(output_frame)

    frame_num += 1

    boxes = sorted(boxes, key=lambda b: b[2] - b[0], reverse=True)[:2]

    player_centers = []
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        foot_x = (x1 + x2) / 2 / scale
        foot_y = y2 / scale
        player_centers.append((foot_x, foot_y))

    if len(player_centers) == 2:
        if player1_ref is None:
            player_centers.sort(key=lambda p: p[0])  
        player1_ref = player_centers[0]  

        d0 = np.linalg.norm(np.array(player_centers[0]) - np.array(player1_ref))
        d1 = np.linalg.norm(np.array(player_centers[1]) - np.array(player1_ref))
        if d0 < d1:
            p1, p2 = player_centers[0], player_centers[1]
        else:
            p1, p2 = player_centers[1], player_centers[0]

        csv_writer.writerow([frame_num, p1[0], p1[1], p2[0], p2[1]])

    # Annotate and write only valid frames
    annotated_frame = results[0].plot()
    output_frame = cv2.resize(annotated_frame, (width, height))
    out.write(output_frame)  


    frame_num += 1

# Release video resources
cap.release()
out.release()
cv2.destroyAllWindows()
# Close the CSV file
csv_file.close()