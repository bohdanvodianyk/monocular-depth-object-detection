from PIL import Image
import src.depth_pro as depth_pro
import cv2
import numpy as np
import glob
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Initialize YOLO and depth models
yolo_model = YOLO("yolo11s.pt")
depth_model, transform = depth_pro.create_model_and_transforms()
depth_model.eval()

# Folder paths
input_folder_path = "data/car_1"  # Folder with input images
output_folder_with_distance = "output/car_1_images_with_distance"  # Folder for images with distance annotations
output_folder_depth_maps = "output/car_1_depth_maps"  # Folder for depth map images

# Create output folders if they don’t exist
os.makedirs(output_folder_with_distance, exist_ok=True)
os.makedirs(output_folder_depth_maps, exist_ok=True)

# Process each image in the input folder
for image_path in glob.glob(os.path.join(input_folder_path, "*.jpg")):
    # Read the input image
    yolo_input = cv2.imread(image_path)

    # YOLO inference for car detection
    results = yolo_model(yolo_input)
    car_boxes = []

    # Extract bounding boxes for cars
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            if result.names[int(cls)] == "car":
                x1, y1, x2, y2 = map(int, box[:4])
                car_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(yolo_input, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Depth prediction
    image, _, f_px = depth_pro.load_rgb(image_path)
    depth_input = transform(image)

    # Run depth model inference
    prediction = depth_model.infer(depth_input, f_px=f_px)
    depth = prediction["depth"]
    depth_np = depth.squeeze().cpu().numpy()

    # Annotate each detected car with depth information
    for x1, y1, x2, y2 in car_boxes:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        depth_value = depth_np[center_y, center_x]

        text = f"Depth: {depth_value:.2f} m"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        # Position the text with a black background for readability
        text_x = x1
        text_y = y1 - 10
        rect_x1 = text_x - 5
        rect_y1 = text_y - text_size[1] - 10
        rect_x2 = text_x + text_size[0] + 5
        rect_y2 = text_y + 5

        cv2.rectangle(yolo_input, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        cv2.putText(yolo_input, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    # Save the annotated image with depth
    image_name = os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_folder_with_distance, f"{os.path.splitext(image_name)[0]}_with_distance.jpg"), yolo_input)

    # Prepare depth map for visualization
    max_depth_visualization = 60
    depth_clipped = np.clip(depth_np, 0, max_depth_visualization)
    depth_min = depth_clipped.min()
    depth_max = max_depth_visualization
    depth_normalized = (depth_clipped - depth_min) / (depth_max - depth_min)

    # Save the depth map as a colored image
    plt.imsave(os.path.join(output_folder_depth_maps, f"{os.path.splitext(image_name)[0]}_depth.png"),
               depth_normalized, cmap=plt.cm.jet)

print("Processing complete. Check output folders for results.")