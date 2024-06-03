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
        print("Blad: Nie mozna otworzyc kamery.")
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
            left_region = output[:, :2 * width // 5]  # 40% szerokosci
            center_region = output[:, 2 * width // 5: 3 * width // 5]  # 20% szerokosci
            right_region = output[:, 3 * width // 5:]  # 40% szerokosci

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
            min_region = min(region_means, key=region_means.get)
            print(f"Region z najmniejsza wartoscia glebokosci: {min_region}")

            # Sprawdzenie, czy średnia wartość w centralnym regionie jest mniejsza niż określony próg
            if center_mean < center_mean_threshold:
                print("Przeszkoda w centralnym regionie.")
                if left_mean > right_mean:
                    direction = "Prawo"
                else:
                    direction = "Lewo"
            else:
                direction = "Do przodu"
            
            # Aktualizacja etykiety kierunku w GUI
            if direction == "Do przodu":
                direction_label.config(text=f"Kierunek: {direction}")
            else:
                direction_label.config(text=f"Kierunek: Skret {direction}")

            # Wizualizacja mapy głębokości
            color_map = plt.get_cmap('jet_r')
            colorful_output = color_map(output / output.max())

            # Wyświetlanie informacji o kierunku na górze ekranu
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3
            color = (255, 255, 255)  # bialy kolor
            if direction != "Do przodu":
                text = f"Skret {direction}"
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (colorful_output.shape[1] - text_size[0]) // 2
                text_y = text_size[1] + 10
                cv2.putText(colorful_output, text, (text_x, text_y), font, font_scale, color, thickness)

            cv2.imshow("Mapa glebokosci", colorful_output)

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
root.title("Interfejs sterowania wozkiem")

# Tworzenie i umieszczanie etykiet i pól tekstowych
tk.Label(root, text="Indeks kamery:").grid(row=0, column=0, padx=5, pady=5)
camera_index_entry = tk.Entry(root)
camera_index_entry.grid(row=0, column=1, padx=5, pady=5)
camera_index_entry.insert(0, "0")  # Domyslny indeks kamery

tk.Label(root, text="Prog sredniej centralnej:").grid(row=1, column=0, padx=5, pady=5)
center_mean_threshold_entry = tk.Entry(root)
center_mean_threshold_entry.grid(row=1, column=1, padx=5, pady=5)
center_mean_threshold_entry.insert(0, str(default_th))  # Domyslny prog wartosci

# Tworzenie i umieszczanie etykiety kierunku
direction_label = tk.Label(root, text="Kierunek: ")
direction_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

# Tworzenie i umieszczanie przycisku start
start_button = tk.Button(root, text="Start", command=start_video_capture)
start_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

# Uruchomienie głównej pętli
root.mainloop()

