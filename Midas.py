import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

# model_type = "DPT_Large"     # MiDaS v3 - Large     (największa dokładność, najwolniejsze działanie)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (średnia dokładność, średnie działanie)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (najmniejsza dokładność, najszybsze działanie)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

default_th = 400

def capture_live_video(camera_index=0, center_mean_threshold=default_th):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Błąd: Nie można otworzyć kamery.")
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

            # Podział obrazu na regiony lewy, centralny i prawy
            height, width = output.shape
            left_region = output[:, :2 * width // 5]  # 40% szerokości
            center_region = output[:, 2 * width // 5: 3 * width // 5]  # 20% szerokości
            right_region = output[:, 3 * width // 5:]  # 40% szerokości

            # Obliczenie średniej wartości głębokości dla każdego regionu
            left_mean = np.mean(left_region)
            center_mean = np.mean(center_region)
            right_mean = np.mean(right_region)

            # Znalezienie regionu z największą średnią wartością głębokości
            region_means = {
                "lewo": left_mean,
                "centrum": center_mean,
                "prawo": right_mean
            }
            max_region = max(region_means, key=region_means.get)
            print(f"Region z największą wartością głębokości: {max_region}")
            min_region = min(region_means, key=region_means.get)
            print(f"Region z najmniejszą wartością głębokości: {min_region}")

            # Sprawdzenie, czy średnia wartość w centralnym regionie jest mniejsza niż określony próg
            if center_mean < center_mean_threshold:
                print("Średnia wartość w regionie centralnym jest mniejsza niż próg")
                direction = "do przodu"
            else:
                if max_region == "lewo":
                    direction = "w prawo"
                elif max_region == "prawo":
                    direction = "w lewo"
                else:
                    direction = "do przodu"
            
            # Aktualizacja etykiety kierunku w GUI
            direction_label.config(text=f"Kierunek: Skręć {min_region.capitalize()}")
            
            # Wizualizacja mapy głębokości
            color_map = plt.get_cmap('jet_r')
            colorful_output = color_map(output / output.max())

            cv2.imshow("Mapa głębokości", colorful_output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def start_video_capture():
    camera_index = int(camera_index_entry.get())
    center_mean_threshold = int(center_mean_threshold_entry.get())
    capture_live_video(camera_index, center_mean_threshold)

# Tworzenie głównego okna
root = tk.Tk()
root.title("Interfejs sterowania wózkiem")

# Tworzenie i umieszczanie etykiet i pól tekstowych
tk.Label(root, text="Indeks kamery:").grid(row=0, column=0, padx=5, pady=5)
camera_index_entry = tk.Entry(root)
camera_index_entry.grid(row=0, column=1, padx=5, pady=5)
camera_index_entry.insert(0, "0")  # Domyślny indeks kamery

tk.Label(root, text="Próg średniej centralnej:").grid(row=1, column=0, padx=5, pady=5)
center_mean_threshold_entry = tk.Entry(root)
center_mean_threshold_entry.grid(row=1, column=1, padx=5, pady=5)
center_mean_threshold_entry.insert(0, str(default_th))  # Domyślny próg wartości

# Tworzenie i umieszczanie etykiety kierunku
direction_label = tk.Label(root, text="Kierunek: ")
direction_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

# Tworzenie i umieszczanie przycisku start
start_button = tk.Button(root, text="Start", command=start_video_capture)
start_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

# Uruchomienie głównej pętli
root.mainloop()

