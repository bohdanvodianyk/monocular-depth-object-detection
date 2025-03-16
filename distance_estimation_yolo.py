from PIL import Image
import src.depth_pro as depth_pro
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

yolo_model = YOLO("yolo11s.pt")

image_path = "data/frame_0163.jpg"

yolo_input = cv2.imread(image_path)

results = yolo_model(yolo_input)

car_boxes = []

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        if result.names[int(cls)] == "car":
            x1, y1, x2, y2 = map(int, box[:4])
            car_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(yolo_input, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Car Detection", yolo_input)
cv2.waitKey(0)
cv2.destroyAllWindows()

depth_model, transform = depth_pro.create_model_and_transforms()
depth_model.eval()

image, _, f_px = depth_pro.load_rgb(image_path)
depth_input = transform(image)

prediction = depth_model.infer(depth_input, f_px=f_px)
depth = prediction["depth"]

depth_np = depth.squeeze().cpu().numpy()

for x1, y1, x2, y2 in car_boxes:
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    depth_value = depth_np[center_y, center_x]

    text = f"Depth: {depth_value:.2f} m"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    text_x = x1
    text_y = y1 - 10
    rect_x1 = text_x - 5
    rect_y1 = text_y - text_size[1] - 10
    rect_x2 = text_x + text_size[0] + 5
    rect_y2 = text_y + 5

    cv2.rectangle(yolo_input, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    cv2.putText(yolo_input, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

cv2.imshow("Car Detection with Depth", yolo_input)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("output/car_detection_with_depth.jpg", yolo_input)

# Manually set the maximum depth value for visualization
max_depth_visualization = 50

# Clip the depth values to the maximum specified
depth_clipped = np.clip(depth, 0, max_depth_visualization)

# Normalize the clipped depth values for visualization
depth_min = depth_clipped.min()
depth_max = max_depth_visualization
depth_normalized = (depth_clipped - depth_min) / (depth_max - depth_min)

# Optionally save the depth as an 8-bit grayscale image with a colormap
plt.imsave(f"output/depth_output.png", depth_normalized.squeeze(), cmap=plt.cm.jet)