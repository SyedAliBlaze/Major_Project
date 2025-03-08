import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import os
import threading
import webbrowser
from ultralytics import YOLO
import logging
from datetime import datetime

# Create directory for screenshots
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

DEFAULT_MODEL = "yolo11s.pt"
def suppress_yolo_logs():
    logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

def restore_logs():
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

class YOLODetector:
    def __init__(self, model_path=DEFAULT_MODEL):
        self.model_path = model_path
        suppress_yolo_logs()  # Suppress logs before loading YOLO
        self.model = self.load_model(model_path)
        restore_logs()  # Restore logs after loading

    def load_model(self, model_path):
        try:
            return YOLO(model_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return None

    def capture_screenshot(self, frame):
        filename = f"screenshots/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, frame)
        # Removed the messagebox.showinfo call to avoid showing the pop-up window

    def resize_frame(self, frame, width=1280, height=720):
        h, w = frame.shape[:2]
        scale = min(width / w, height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h))
        top = (height - new_h) // 2
        bottom = height - new_h - top
        left = (width - new_w) // 2
        right = width - new_w - left
        return cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT)

    def detect_image(self, image_path):
        if not self.model:
            messagebox.showerror("Error", "No model loaded!")
            return
        img = cv2.imread(image_path)
        img = self.resize_frame(img)
        results = self.model(img)
        for r in results:
            img = r.plot()
        cv2.imshow("Object Detection - Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_video(self, video_path):
        if not self.model:
            messagebox.showerror("Error", "No model loaded!")
            return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file!")
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.resize_frame(frame)
            results = self.model(frame)
            for r in results:
                frame = r.plot()
            cv2.imshow("Object Detection - Video", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.capture_screenshot(frame)
        cap.release()
        cv2.destroyAllWindows()

    def detect_webcam(self, camera_index=0):
        if not self.model:
            messagebox.showerror("Error", "No model loaded!")
            return
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open webcam!")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.resize_frame(frame)
            flipped_frame = cv2.flip(frame, 1)  # Flip the frame horizontally
            results = self.model(flipped_frame)
            for r in results:
                flipped_frame = r.plot()
            cv2.imshow("Object Detection - Webcam", flipped_frame)  # Display the flipped frame directly
            cv2.moveWindow("Object Detection - Webcam", 0, 0)  # Move window to top-left corner
            cv2.setWindowProperty("Object Detection - Webcam", cv2.WND_PROP_TOPMOST, 1)  # Bring window to front
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.capture_screenshot(flipped_frame)  # Capture the flipped frame
        cap.release()
        cv2.destroyAllWindows()

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection App")
        self.root.geometry("500x400")
        self.detector = YOLODetector()
        self.model_name = tk.StringVar()
        self.model_name.set(f"Model: {DEFAULT_MODEL}")
        self.camera_index = tk.StringVar(value="Camera 0")  # Default to system camera
        self.main_menu()

    def main_menu(self):
        self.clear_window()
        tk.Label(self.root, text="Main Menu", font=("Arial", 16, "bold")).pack(pady=20)

        tk.Button(self.root, text="Object Detection", width=25, height=2, command=self.object_detection_menu).pack(pady=10)
        tk.Button(self.root, text="Settings", width=25, height=2, command=self.settings_menu).pack(pady=10)
        tk.Button(self.root, text="View Screenshots", width=25, height=2, command=self.view_screenshots).pack(pady=10)
        tk.Button(self.root, text="Quit", width=25, height=2, bg="red", fg="white", command=self.root.quit).pack(pady=20)
        
        info_button = tk.Button(self.root, text="ℹ Info", font=("Arial", 10, "bold"), bg="gray", fg="white", command=self.show_info)
        info_button.place(x=10, y=360)

    def object_detection_menu(self):
        self.clear_window()
        self.root.update_idletasks()  # Force an update to ensure widgets are displayed properly

        tk.Label(self.root, text="Object Detection", font=("Arial", 14, "bold")).pack(pady=10)

        tk.Button(self.root, text="Detect Image", width=25, command=self.detect_image).pack(pady=5)
        tk.Button(self.root, text="Detect Video", width=25, command=self.detect_video).pack(pady=5)
        tk.Button(self.root, text="Detect Webcam", width=25, command=self.detect_webcam).pack(pady=5)

        tk.Label(self.root, text="Select Camera:", font=("Arial", 12)).pack(pady=5)
        self.camera_dropdown = ttk.Combobox(self.root, state="readonly", textvariable=self.camera_index)
        self.camera_dropdown.pack(pady=5)
        self.scan_cameras()

        tk.Button(self.root, text="Back", width=25, bg="gray", command=self.main_menu).pack(pady=20)

    def scan_cameras(self):
        camera_options = []
        for i in range(10):  # Scan for up to 10 cameras
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_options.append(f"Camera {i}")
                cap.release()
        self.camera_dropdown['values'] = camera_options
        if camera_options:
            self.camera_dropdown.current(0)  # Set default selection to the first available camera

    def detect_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            self.detector.detect_image(file_path)

    def detect_video(self):
        file_path = filedialog.askopenfilename(title="Select a Video", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if file_path:
            threading.Thread(target=self.detector.detect_video, args=(file_path,)).start()

    def detect_webcam(self):
        camera_index = int(self.camera_index.get().split()[-1])  # Extract the integer part
        threading.Thread(target=self.detector.detect_webcam, args=(camera_index,)).start()

    def view_screenshots(self):
        os.startfile(os.path.abspath("screenshots"))

    def show_info(self):
        info_text = (
            "📌 How to Use Object Detection App\n\n"
            "1️⃣ **Detect Objects in Images**:\n"
            "   - Click 'Object Detection' → 'Detect Image'.\n"
            "   - Select an image file (.jpg, .png, .jpeg).\n"
            "   - The detected objects will be displayed.\n\n"
            
            "2️⃣ **Detect Objects in Videos**:\n"
            "   - Click 'Object Detection' → 'Detect Video'.\n"
            "   - Select a video file (.mp4, .avi, .mov).\n"
            "   - Object detection runs in real time.\n"
            "   - Press 'Q' to quit, 'S' to take a screenshot.\n\n"
            
            "3️⃣ **Live Object Detection (Webcam)**:\n"
            "   - Click 'Object Detection' → 'Detect Webcam'.\n"
            "   - Select the camera from the dropdown menu.\n"
            "   - Object detection runs using your selected webcam.\n"
            "   - Press 'Q' to quit, 'S' to take a screenshot.\n\n"
            
            "4️⃣ **Load a Custom Model**:\n"
            "   - Go to 'Settings' → 'Load Model'.\n"
            "   - Select a trained YOLO model (.pt file).\n"
            "   - The new model will be loaded for detection.\n\n"
            
            "5️⃣ **View Screenshots**:\n"
            "   - Click 'View Screenshots' to open the saved images.\n"
            "   - Screenshots are saved in the 'screenshots' folder.\n\n"
            
            "⚠ Note: Ensure you have a trained YOLO model loaded before running detection.\n"
            "Developed by **Syed Ali N.** 🚀"
        )
        
        messagebox.showinfo("Application Info", info_text)

    def settings_menu(self):
        self.clear_window()
        tk.Label(self.root, text="Settings", font=("Arial", 14, "bold")).pack(pady=10)

        tk.Label(self.root, textvariable=self.model_name, font=("Arial", 12, "bold"), fg="blue").pack(pady=5)
        tk.Button(self.root, text="Load Model", width=25, command=self.load_model).pack(pady=5)
        tk.Button(self.root, text="Gather Dataset", width=25, command=lambda: webbrowser.open("https://universe.roboflow.com/")).pack(pady=5)
        tk.Button(self.root, text="Train Model", width=25, command=lambda: webbrowser.open("https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb")).pack(pady=5)
        tk.Button(self.root, text="Back", width=25, bg="gray", command=self.main_menu).pack(pady=20)

    def load_model(self):
        file_path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=[("Model Files", "*.pt")])
        if file_path:
            self.detector = YOLODetector(file_path)
            self.model_name.set(f"Model: {os.path.basename(file_path)}")
            messagebox.showinfo("Success", "Model loaded successfully!")

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
