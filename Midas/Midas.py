
import cv2
import torch
import urllib.request

import matplotlib.pyplot as plt


model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# cap = cv2.VideoCapture(0) #zrodlo video
def capture_live_video(camera_index=0):
    # Inicjalizacja kamery
    cap = cv2.VideoCapture(camera_index)

    # Sprawdzenie, czy kamera została poprawnie otwarta
    if not cap.isOpened():
        print("Błąd: Nie można otworzyć kamery.")
        return

    try:
        while True:
            ret, frame = cap.read()
            img = frame
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # zmiana na bgr

            input_batch = transform(img).to(device)
            with torch.no_grad():
                prediction = midas(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(0),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            output = prediction.cpu().numpy()

            color_map = plt.get_cmap('jet')  # mozna pokolorowac dodatkowo
            colorful_output = color_map(output / output.max())

            cv2.imshow("wynik", colorful_output)

            # Przerwanie pętli, jeśli naciśnięto klawisz 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Zwolnienie zasobów po zakończeniu
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    # Uruchomienie funkcji, domyślnie z indeksem kamery 0
    capture_live_video()