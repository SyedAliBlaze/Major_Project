import tkinter as tk
from tkinter import filedialog
import os
import subprocess
import webbrowser

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection & Training")
        self.root.geometry("400x300")
        
        self.selected_model = tk.StringVar()
        
        self.main_menu()

    def main_menu(self):
        self.clear_window()
        
        self.object_detection_btn = tk.Button(self.root, text="Object Detection", command=self.object_detection_menu)
        self.object_detection_btn.pack(pady=10)
        
        self.train_model_btn = tk.Button(self.root, text="Train Model", command=self.train_model_menu)
        self.train_model_btn.pack(pady=10)

    def object_detection_menu(self):
        self.clear_window()
        
        self.back_btn = tk.Button(self.root, text="⬅ Back", command=self.main_menu)
        self.back_btn.pack(anchor='nw', padx=5, pady=5)
        
        self.live_feed_btn = tk.Button(self.root, text="Live Feed")
        self.live_feed_btn.pack(pady=5)
        
        self.video_btn = tk.Button(self.root, text="Video")
        self.video_btn.pack(pady=5)
        
        self.image_btn = tk.Button(self.root, text="Image")
        self.image_btn.pack(pady=5)
    
    def train_model_menu(self):
        self.clear_window()
        
        self.back_btn = tk.Button(self.root, text="⬅ Back", command=self.main_menu)
        self.back_btn.pack(anchor='nw', padx=5, pady=5)
        
        self.create_dataset_btn = tk.Button(self.root, text="Create Dataset", command=self.open_roboflow)
        self.create_dataset_btn.pack(pady=5)
        
        self.train_model_btn = tk.Button(self.root, text="Train Model", command=self.open_colab)
        self.train_model_btn.pack(pady=5)

        # Model Selection
        self.select_model_btn = tk.Button(self.root, text="Select Model", command=self.select_model)
        self.select_model_btn.pack(pady=10)
        
        self.selected_model_label = tk.Label(self.root, textvariable=self.selected_model)
        self.selected_model_label.pack()

        # Program Button
        self.program_btn = tk.Button(self.root, text="Program", command=self.open_yolo_detect)
        self.program_btn.pack(pady=10)

    def select_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.h5;*.pt;*.pth")])
        if file_path:
            self.selected_model.set(file_path)

    def open_yolo_detect(self):
        file_path = "yolo_detect.py"  # File is in the same directory
        if os.path.exists(file_path):
            subprocess.run(["notepad", file_path])  # Opens in Notepad
        else:
            print("File not found!")  # Debugging message

    def open_roboflow(self):
        webbrowser.open("https://universe.roboflow.com/")  # Opens Roboflow Universe

    def open_colab(self):
        webbrowser.open("https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb")  # Opens Google Colab notebook

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
