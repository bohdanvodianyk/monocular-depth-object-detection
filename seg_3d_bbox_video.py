import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from ultralytics import YOLO
import src.depth_pro as depth_pro
import json


# Function to create a point cloud from a depth map and RGB image
def create_full_point_cloud(depth_map, rgb_image, fx, fy, cx, cy):
    full_point_cloud = o3d.geometry.PointCloud()
    points, colors = [], []

    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            z = depth_map[i, j]
            if 0 < z <= 60:  # Limit the depth range for filtering
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
                if 0 < z <= 60:  # Filter by depth range
                    x = (j - cx) * z / fx
                    y = (i - cy) * z / fy
                    points.append([x, y, z])
                    color = rgb_image[i, j] / 255.0  # Normalize color
                    colors.append(color)

    # Convert points and colors to numpy arrays
    points = np.array(points, dtype=np.float64).reshape(-1, 3)
    colors = np.array(colors, dtype=np.float64).reshape(-1, 3)

    # Check if any points were added, if not, skip the point cloud creation
    if points.shape[0] == 0:
        print("No valid points found for the mask. Skipping point cloud creation.")
        return None

    # Set points and colors to point cloud
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Optional: Filter noise from the segmented point cloud
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=1000, std_ratio=0.25)
    return point_cloud.select_by_index(ind)


# Load camera calibration parameters
calibration_data = np.load('CalibrationMatrix_college_cpt.npz')
camera_matrix = calibration_data['Camera_matrix']
fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]

# Load YOLO and depth models outside the loop
yolo_model = YOLO("yolo11s-seg.pt").to("cuda" if torch.cuda.is_available() else "cpu")
depth_model, transform = depth_pro.create_model_and_transforms()
depth_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# Define paths for input images
video_files = sorted(glob.glob("data/car_1/*.jpg"))

# Initialize Open3D visualizer outside the loop
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)

for img_path in video_files:
    print(f"Start processing {img_path}")
    # Load the image
    yolo_input = cv2.imread(img_path)

    # YOLO detection for car segmentation
    results = yolo_model(yolo_input)
    car_masks = []
    for result in results:
        classes = result.boxes.cls.cpu().numpy()
        masks = result.masks.data.cpu().numpy() if hasattr(result, 'masks') else None
        for i, cls in enumerate(classes):
            if result.names[int(cls)] == "car" and masks is not None:
                car_masks.append(masks[i])

    # Depth prediction
    image_rgb, _, f_px = depth_pro.load_rgb(img_path)
    depth_input = transform(image_rgb).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        prediction = depth_model.infer(depth_input, f_px=f_px)
    depth_map = prediction["depth"].cpu().numpy().squeeze()

    # Create the main point cloud
    main_point_cloud = create_full_point_cloud(depth_map, yolo_input, fx, fy, cx, cy)

    # Clear geometries to avoid memory overflow
    vis.clear_geometries()

    # Add main point cloud and bounding boxes to visualizer
    vis.add_geometry(main_point_cloud)
    car_bboxes = []

    for car_mask in car_masks:
        car_point_cloud = create_segmented_point_cloud_for_mask(depth_map, yolo_input, car_mask, fx, fy, cx, cy)

        # Check if car_point_cloud is None
        if car_point_cloud is None:
            print("Skipped a mask as no valid points were found for the point cloud.")
            continue  # Skip to the next mask if no valid point cloud was created

        # Create bounding box for the valid point cloud
        bbox = car_point_cloud.get_minimal_oriented_bounding_box()
        # bbox = car_point_cloud.get_axis_aligned_bounding_box()
        bbox.color = (0, 0, 1)  # Blue bounding box
        car_bboxes.append(bbox)
        vis.add_geometry(bbox)

    # Set the camera view (as before)
    ctr = vis.get_view_control()
    ctr.set_front([-0.45, 0.75, -1.5])
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_up([0.0, -1.0, -0.1])
    ctr.set_zoom(0.005)

    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)

    # Convert the captured image for saving
    image_np = (255 * np.asarray(image)).astype(np.uint8)
    frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    file_name = img_path.replace("\\", "/").split("/")[-1].split(".")[0]
    plt.imsave(f"output/car_1/{file_name}_pcd_bbox.png", frame_bgr)
    print(f"{file_name} done...")

# Close the visualizer after processing all frames
vis.destroy_window()
print("Images saved successfully.")