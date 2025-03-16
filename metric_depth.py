import numpy as np
import open3d as o3d
import torch
import src.depth_pro as depth_pro
from PIL import Image
import matplotlib.pyplot as plt

def depth_map_to_colored_point_cloud(depth_map, rgb_image, camera_matrix):
    # Create a point cloud object
    point_cloud = o3d.geometry.PointCloud()

    # Extract focal lengths and principal point from the camera matrix
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # Initialize an empty list to store points and colors
    points = []
    colors = []

    # Get the height and width of the depth map
    height, width = depth_map.shape

    # Iterate through each pixel in the depth map
    for i in range(height):
        for j in range(width):
            z = depth_map[i, j]
            if 0 < z <= 50:  # Filter with Z maximum 50 (or adjust as needed)
                # Convert pixel coordinates to 3D camera coordinates
                x = (j - cx) * z / fx
                y = (i - cy) * z / fy
                points.append([x, y, z])

                # Get the color from the RGB image
                color = rgb_image[i, j] / 255.0  # Normalize to [0, 1]
                colors.append(color)

    # Convert points and colors to numpy arrays
    points = np.array(points)
    colors = np.array(colors)

    # Set the points and colors in the point cloud
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Path to your image file
image_path = "./data/frame_0000.jpg"

# Load and preprocess the input image
image_rgb, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image_rgb)

# Run inference to get the depth map
with torch.no_grad():
    prediction = model.infer(image, f_px=f_px)

# Extract the depth map and convert it to a NumPy array
depth = prediction["depth"].cpu().numpy().squeeze()

# Load the camera calibration parameters
calibration_data = np.load('CalibrationMatrix_college_cpt.npz')
camera_matrix = calibration_data['Camera_matrix']

file_name = image_path.split("/")[-1].split(".")[0]
# Create the point cloud using the camera matrix
point_cloud = depth_map_to_colored_point_cloud(depth, image_rgb, camera_matrix)
# Estimate normals (optional but can help in outlier removal)
point_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=1000))
# Remove outliers using statistical outlier removal
# Adjust nb_neighbors and std_ratio for different results
cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.5)
point_cloud = point_cloud.select_by_index(ind)
# Optionally, save the point cloud to a file
o3d.io.write_point_cloud(f"output/output_pcd_{file_name}.ply", point_cloud)

# Manually set the maximum depth value for visualization
max_depth_visualization = 50

# Clip the depth values to the maximum specified
depth_clipped = np.clip(depth, 0, max_depth_visualization)

# Normalize the clipped depth values for visualization
depth_min = depth_clipped.min()
depth_max = max_depth_visualization
depth_normalized = (depth_clipped - depth_min) / (depth_max - depth_min)

# Optionally save the depth as an 8-bit grayscale image with a colormap
plt.imsave(f"output/depth_output_{file_name}.png", depth_normalized.squeeze(), cmap=plt.cm.jet)
