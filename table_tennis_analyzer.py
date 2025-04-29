# table_tennis_analyzer.py

from ultralytics import YOLO
import cv2

model = YOLO('yolov8n-pose.pt') 

image_path = 'table-tennis.png'  
image = cv2.imread(image_path)

scale = 0.55
new_width = int(image.shape[1] * scale)
new_height = int(image.shape[0] * scale)

resized_image = cv2.resize(image, (new_width, new_height))
results = model(resized_image, imgsz=1500, conf=0.25)


cv2.imshow("Pose Detection", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
