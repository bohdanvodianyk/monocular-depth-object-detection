import torch
import depth_pro
import matplotlib.pyplot as plt

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Path to your image file
image_path = "./data/frame_0163.jpg"

image_rgb, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image_rgb)

# Run inference to get the depth map
with torch.no_grad():
    prediction = model.infer(image, f_px=f_px)

# Extract the depth map and convert it to a NumPy array
depth = prediction["depth"].cpu().numpy().squeeze()

depth_min = depth.min()
depth_max = depth.max()
depth_normalized = (depth - depth_min) / (depth_max - depth_min)

# Optionally save the depth as an 8-bit grayscale image
plt.imsave(f"output/depth_output.png", depth_normalized.squeeze(), cmap=plt.cm.jet)