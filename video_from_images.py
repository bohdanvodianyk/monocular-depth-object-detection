import cv2
import os
import glob

# Path to the folder containing images
image_folder_path = "data/car_1/"
output_video_path = "output/original.mp4"
frame_rate = 15  # Frames per second for the video

# Get list of image files with .png extension and sort them
images = sorted(glob.glob(os.path.join(image_folder_path, "*.jpg")))

# Check if images were found
if not images:
    raise ValueError("No images found in the specified folder.")

# Read the first image to get dimensions
first_image = cv2.imread(images[0])
height, width, layers = first_image.shape

# Define the video writer with codec and parameters
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 output or 'XVID' for .avi
video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

# Loop through each image and add it to the video
for image_path in images:
    img = cv2.imread(image_path)
    video.write(img)  # Add frame to video

# Release the video writer
video.release()
print("Video created successfully:", output_video_path)