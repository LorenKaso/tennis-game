# table_tennis_analyzer.py

from ultralytics import YOLO
import cv2

model = YOLO('yolov8n-pose.pt') 

image_path = 'table-tennis.png'  
image = cv2.imread(image_path)

results = model(image)

for result in results:
    result.plot()  

cv2.imshow("Pose Detection", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
