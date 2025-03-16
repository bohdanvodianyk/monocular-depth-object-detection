import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import open3d as o3d
from ultralytics import YOLO
import src.depth_pro as depth_pro

# Load the YOLO model and depth model
yolo_model = YOLO("yolo11s-seg.pt")
depth_model, transform = depth_pro.create_model_and_transforms()
depth_model.eval()

# Path to the input image
image_path = "data/frame_0163.jpg"
yolo_input = cv2.imread(image_path)

# YOLO prediction for car detection and segmentation
results = yolo_model(yolo_input)
car_masks = []

for result in results:
    classes = result.boxes.cls.cpu().numpy()
    masks = result.masks.data.cpu().numpy() if hasattr(result, 'masks') else None

    for i, cls in enumerate(classes):
        if result.names[int(cls)] == "car" and masks is not None:
            car_masks.append(masks[i])

# Depth prediction for the input image
image_rgb, _, f_px = depth_pro.load_rgb(image_path)
depth_input = transform(image_rgb)
with torch.no_grad():
    prediction = depth_model.infer(depth_input, f_px=f_px)
depth_map = prediction["depth"].cpu().numpy().squeeze()

# Load camera calibration parameters
calibration_data = np.load('CalibrationMatrix_college_cpt.npz')
camera_matrix = calibration_data['Camera_matrix']
fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]


# Function to create a point cloud from a depth map and RGB image
def create_full_point_cloud(depth_map, rgb_image, fx, fy, cx, cy):
    full_point_cloud = o3d.geometry.PointCloud()
    points, colors = [], []

    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            z = depth_map[i, j]
            if 0 < z <= 50:  # Limit the depth range for filtering
                x = (j - cx) * z / fx
                y = (i - cy) * z / fy
                points.append([x, y, z])
                color = rgb_image[i, j] / 255.0  # Normalize color
                colors.append(color)

    # Set points and colors to full point cloud
    full_point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    full_point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    cl, ind = full_point_cloud.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.25)
    return full_point_cloud.select_by_index(ind)


# Generate the full point cloud for the entire scene
main_point_cloud = create_full_point_cloud(depth_map, yolo_input, fx, fy, cx, cy)


# Function to create a segmented point cloud for a given mask
def create_segmented_point_cloud_for_mask(depth_map, rgb_image, mask, fx, fy, cx, cy):
    point_cloud = o3d.geometry.PointCloud()
    points, colors = [], []

    # Resize mask to match depth and RGB image dimensions
    mask_resized = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]))
    mask_bool = mask_resized > 0.5  # Threshold for binary mask

    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            if mask_bool[i, j]:  # Only process points within mask
                z = depth_map[i, j]
                if 0 < z <= 50:  # Filter by depth range
                    x = (j - cx) * z / fx
                    y = (i - cy) * z / fy
                    points.append([x, y, z])
                    color = rgb_image[i, j] / 255.0  # Normalize color
                    colors.append(color)

    # Set points and colors to point cloud
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))

    # Optional: Filter noise from the segmented point cloud
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=1000, std_ratio=0.25)
    return point_cloud.select_by_index(ind)


# Process each car mask individually to create a point cloud and bounding box
car_point_clouds = []
car_bboxes = []

for idx, car_mask in enumerate(car_masks):
    # Create point cloud for each car
    car_point_cloud = create_segmented_point_cloud_for_mask(depth_map, yolo_input, car_mask, fx, fy, cx, cy)

    # Calculate axis-aligned bounding box for the car
    bbox = car_point_cloud.get_minimal_oriented_bounding_box()
    bbox.color = (1, 0, 0)  # Red color for bounding box

    # Append to lists for visualization and saving
    car_point_clouds.append(car_point_cloud)
    car_bboxes.append(bbox)

# Visualize the full point cloud with all cars' bounding boxes overlaid
# o3d.visualization.draw_geometries([main_point_cloud] + car_bboxes)

# Create a visualizer and add the main point cloud and car bounding boxes
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(main_point_cloud)
for bbox in car_bboxes:
    vis.add_geometry(bbox)

# Set a specific camera viewpoint
ctr = vis.get_view_control()
ctr.set_front([-0.45, 0.75, -1.5])  # Set this to the desired view direction
ctr.set_lookat([0.0, 0.0, 0.0])  # Set this to the focus point
ctr.set_up([0.0, -1.0, -0.1])     # Set this to the upward direction
ctr.set_zoom(0.005)                 # Adjust the zoom level as needed

# Capture the image from the viewpoint
vis.poll_events()
vis.update_renderer()
image = vis.capture_screen_float_buffer(do_render=True)
vis.destroy_window()

# Convert the captured image to a format that can be saved
image_np = (255 * np.asarray(image)).astype(np.uint8)
image_pil = Image.fromarray(image_np)
image_pil.show()
plt.imsave("output/point_cloud_view.png", image_pil)
# image_pil.save("output/point_cloud_view.png")

