import Midas.midas as midas
import tkinter as tk
from tkinter import ttk
from EEG.predict import prediction
import threading
from collections import Counter
direction = "Prosto"
prediction_label = 0
center_mean_threshold = 400

class LimitedArray:
    def __init__(self, limit):
        self.limit = limit
        self.array = [0] * limit

    def append_start(self, element):
        if len(self.array) < self.limit:
            self.array = [element] + self.array
        else:
            self.array = [element] + self.array[:-1]
    def most_common(self):
        if self.array:
            counter = Counter(self.array)
            most_common_value = counter.most_common(1)[0][0]
            return most_common_value
        else:
            print("Array is empty.")   
            
label_buff = LimitedArray(3)      
        
    
def EEGloopSIM(window=5, hop=0.2, t=0):
    global prediction_label
    # Fetch the prediction label
    prediction_label = prediction(t, t + window)[0]
    # Update the label text
    prediction_label_var.set(f'EEGTEST: {prediction_label}')
    # Increment t by hop
    t += hop
    if t > 116:
        t = 0
        
    # Schedule the next call
    root.after(2000, lambda: EEGloopSIM(window, hop, t))
    
    
def check_midas():
    # left_label_var.set(f'Lewo: {midas.left}')
    left_label_var.set(f'Lewo: {int(midas.left)}')
    center_label_var.set(f'Centrum: {int(midas.center)}')
    right_label_var.set(f'Prawo: {int(midas.right)}')
    
    direction_label_var.set(f'Kierunek Jazdy: {direction}')
    
    
    root.after(10, lambda: check_midas())
    
    
def start_video_capture():
    global center_mean_threshold
    camera_index = int(camera_index_entry.get())
    center_mean_threshold = int(center_mean_threshold_entry.get())
    
    capture_thread = threading.Thread(target=midas.capture_live_video, args=(center_mean_threshold,))
    capture_thread.start()
    
    root.after(100, lambda: onedirection())
    root.after(100, lambda: check_midas())
    
    
    
direction_map = {
    1 :"Prosto",
    2 :"Prawo",
    3 : "Lewo",
    0 : "Stop"
}    

def onedirection():
    global prediction_label, center_mean_threshold, direction, label_buff
    th = center_mean_threshold
    
    label_buff.append_start(int(prediction_label))
    pred = label_buff.most_common()
    
    ml = midas.left
    mc = midas.center
    mr = midas.right
    if pred == 1:  # jazda prosto
        if mc < th:
            direction = direction_map[pred]
        elif mr < mc and mr < ml:
            direction = direction_map[2]
        elif ml < mc and ml < mr:
            direction = direction_map[3]
        else:
            direction = direction_map[0]
    elif pred == 2:
            direction = direction_map[pred]
    elif pred == 3:
            direction = direction_map[pred]       
    else:
            direction = direction_map[0]       
    root.after(100,lambda: onedirection())
if __name__ == "__main__":
    
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
    center_mean_threshold_entry.insert(0, str(midas.default_th))  # Default threshold value

    # Create and place the prediction label
    prediction_label_var = tk.StringVar()
    prediction_label_var.set('EEGTEST: 0')
    prediction_label_display = tk.Label(root, textvariable=prediction_label_var)
    prediction_label_display.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
    
    
    
    direction_label_var= tk.StringVar()
    direction_label_var.set('Kierunek Jazdy: ')
    direction_label_display= tk.Label(root, textvariable=direction_label_var)
    direction_label_display.grid(row=4, column=0, columnspan=2, padx=5, pady=5)



    left_label_var= tk.StringVar()
    left_label_var.set('Lewo: 0')
    left_label_display = tk.Label(root, textvariable=left_label_var)
    left_label_display.grid(row=6, column=0, columnspan=2, padx=5, pady=5)
    


    center_label_var= tk.StringVar()
    center_label_var.set('Centrum: 0')
    center_label_display = tk.Label(root, textvariable=center_label_var)
    center_label_display.grid(row=7, column=0, columnspan=2, padx=5, pady=5)



    right_label_var= tk.StringVar()
    right_label_var.set('Prawo: 0')
    right_label_display = tk.Label(root, textvariable=right_label_var)
    right_label_display.grid(row=8, column=0, columnspan=2, padx=5, pady=5)
    
    
    
    # Create and place the start button
    start_button = tk.Button(root, text="Start", command=start_video_capture)
    start_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
    
    
    # Start the EEG loop simulation
    root.after(1000, lambda: EEGloopSIM())

    # Run the main loop
    root.mainloop()
