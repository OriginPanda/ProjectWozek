import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

left = 0
center = 0
right = 0
default_th = 600

def capture_live_video(camera_index=0, center_mean_threshold=default_th):
    global left, center, right  # Declare global variables
    
    cap = cv2.VideoCapture(0)
 
    if not cap.isOpened():
        print("Error: Cannot open the camera.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = transform(img).to(device)
            with torch.no_grad():
                prediction = midas(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            output = prediction.cpu().numpy()

            # Divide the image into left, center, and right regions
            height, width = output.shape
            left_region = output[:, :2 * width // 5]  # 40% width
            center_region = output[:, 2 * width // 5: 3 * width // 5]  # 20% width
            right_region = output[:, 3 * width // 5:]  # 40% width

            # Calculate mean depth value for each region
            left_mean = np.mean(left_region)
            center_mean = np.mean(center_region)
            right_mean = np.mean(right_region)

            left = left_mean
            center = center_mean
            right = right_mean

            # Find the region with the highest mean depth value
            region_means = {
                "left": left_mean,
                "center": center_mean,
                "right": right_mean
            }
            # max_region = max(region_means, key=region_means.get)
            # print(f"Most depth value region: {max_region}")
            # min_region = min(region_means, key=region_means.get)
            # print(f"Least depth value region: {min_region}")
            # Check if center mean is less than the specified threshold
            if center_mean < center_mean_threshold:
                
                print("free ahead")
                # Perform some action here if needed
        
            # Visualize the depth map
            color_map = plt.get_cmap('jet_r')
            colorful_output = color_map(output / output.max())

            # Overlay mean values on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            color = (255, 255, 255)  # white color

            cv2.imshow("Kamera", colorful_output)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
