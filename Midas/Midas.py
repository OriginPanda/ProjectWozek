import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from EEG.predict import prediction

# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

default_th = 400;
prediction_label = 0
def capture_live_video(camera_index=0, center_mean_threshold=default_th):
    left = 0
    center = 0
    right = 0
    
    cap = cv2.VideoCapture(camera_index)
 
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

            # Find the region with the highest mean depth value
            region_means = {
                "left": left_mean,
                "center": center_mean,
                "right": right_mean
            }
            max_region = max(region_means, key=region_means.get)
            print(f"Most depth value region: {max_region}")
            min_region = min(region_means, key=region_means.get)
            print(f"Least depth value region: {min_region}")
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

            cv2.putText(colorful_output, f'Left Mean: {left_mean:.2f}', (10, 30), font, font_scale, color, thickness)
            cv2.putText(colorful_output, f'Center Mean: {center_mean:.2f}', (10, 60), font, font_scale, color, thickness)
            cv2.putText(colorful_output, f'Right Mean: {right_mean:.2f}', (10, 90), font, font_scale, color, thickness)
            cv2.putText(colorful_output, f'Most Depth: {max_region}', (10, 120), font, font_scale, color, thickness)
            cv2.putText(colorful_output, f'Least Depth: {min_region}', (10, 150), font, font_scale, color, thickness)
            cv2.imshow("Depth Map", colorful_output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
    return left , center ,right

def EEGloopSIM(hop):
    prediction_label = prediction(hop,hop+4.5)
    
    print('EEGTEST: ' + prediction_label)
    
    root.after(5000, EEGloopSIM(hop+4.5))
    
    
def start_video_capture():
    camera_index = int(camera_index_entry.get())
    center_mean_threshold = int(center_mean_threshold_entry.get())
    capture_live_video(camera_index, center_mean_threshold)

# Create the main window
root = tk.Tk()
root.title("Main app")

# Create and place labels and entry widgets
tk.Label(root, text="Camera Index:").grid(row=0, column=0, padx=5, pady=5)
camera_index_entry = tk.Entry(root)
camera_index_entry.grid(row=0, column=1, padx=5, pady=5)
camera_index_entry.insert(0, "0")  # Default camera index

tk.Label(root, text="Center Mean Threshold:").grid(row=1, column=0, padx=5, pady=5)
center_mean_threshold_entry = tk.Entry(root)
center_mean_threshold_entry.grid(row=1, column=1, padx=5, pady=5)
center_mean_threshold_entry.insert(0, str(default_th))  # Default threshold value

# Create and place the start button
start_button = tk.Button(root, text="Start", command=start_video_capture)
start_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

root.after(0,EEGloopSIM())
# Run the main loop
root.mainloop()
