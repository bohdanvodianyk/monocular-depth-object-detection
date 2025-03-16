from PIL import Image
import src.depth_pro as depth_pro
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO model
yolo_model = YOLO("yolo11s-seg.pt")

# Path to the input image
image_path = "data/frame_0163.jpg"

# Read the input image using OpenCV
yolo_input = cv2.imread(image_path)

# Run the YOLO model to get predictions
results = yolo_model(yolo_input)

# Initialize lists to hold car bounding boxes and segmentation masks
car_boxes = []
segmentation_masks = []

# Process each result
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    masks = result.masks.data.cpu().numpy() if hasattr(result, 'masks') else None

    # Process each detection
    for i, (box, cls) in enumerate(zip(boxes, classes)):
        if result.names[int(cls)] == "car":
            x1, y1, x2, y2 = map(int, box[:4])
            car_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(yolo_input, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # If segmentation masks are available, add the corresponding mask
            if masks is not None:
                mask = masks[i]
                segmentation_masks.append(mask)

# Display bounding boxes and segmentation masks
for mask in segmentation_masks:
    # Resize the mask to the input image size
    mask_resized = cv2.resize(mask, (yolo_input.shape[1], yolo_input.shape[0]))
    mask_colored = np.zeros_like(yolo_input)
    mask_colored[mask_resized > 0.5] = (0, 0, 255)  # Set mask color to red

    # Overlay mask on the input image
    yolo_input = cv2.addWeighted(yolo_input, 1, mask_colored, 0.5, 0)

cv2.imshow("Car Detection with Segmentation Masks", yolo_input)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Depth model initialization
depth_model, transform = depth_pro.create_model_and_transforms()
depth_model.eval()

# Load the image and preprocess for depth model
image, _, f_px = depth_pro.load_rgb(image_path)
depth_input = transform(image)

# Get depth prediction
prediction = depth_model.infer(depth_input, f_px=f_px)
depth = prediction["depth"]

depth_np = depth.squeeze().cpu().numpy()

# Annotate depth on each detected car
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

# Display the result with depth annotations
cv2.imshow("Car Detection with Depth and Segmentation Masks", yolo_input)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("output/car_detection_with_depth_and_masks.jpg", yolo_input)

# Visualize and save the depth map with a colormap
max_depth_visualization = 50
depth_clipped = np.clip(depth, 0, max_depth_visualization)
depth_min = depth_clipped.min()
depth_max = max_depth_visualization
depth_normalized = (depth_clipped - depth_min) / (depth_max - depth_min)
plt.imsave("output/depth_output.png", depth_normalized.squeeze(), cmap=plt.cm.jet)