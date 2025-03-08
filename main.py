import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import threading
import webbrowser
from ultralytics import YOLO
from datetime import datetime

# Create directory for screenshots
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

class YOLODetector:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        try:
            return YOLO(model_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return None

    def capture_screenshot(self, frame):
        filename = f"screenshots/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, frame)
        messagebox.showinfo("Screenshot Saved", f"Screenshot saved as {filename}")

    def resize_frame(self, frame):
        """ Resize frame while maintaining aspect ratio and halving the horizontal resolution """
        h, w, _ = frame.shape
        new_w = w // 2  # Halve horizontal resolution
        aspect_ratio = h / w
        new_h = int(new_w * aspect_ratio)  # Maintain aspect ratio
        return cv2.resize(frame, (new_w, new_h))

    def detect_image(self, image_path):
        if not self.model:
            messagebox.showerror("Error", "No model loaded!")
            return
        img = cv2.imread(image_path)
        results = self.model(img)
        for r in results:
            img = r.plot()
        img_resized = self.resize_frame(img)  # Apply resizing only in Image Mode
        cv2.imshow("Object Detection - Image", img_resized)
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
            results = self.model(frame)
            for r in results:
                frame = r.plot()
            frame_resized = self.resize_frame(frame)  # Apply resizing for video mode
            cv2.imshow("Object Detection - Video", frame_resized)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.capture_screenshot(frame_resized)
        cap.release()
        cv2.destroyAllWindows()

    def detect_webcam(self):
        if not self.model:
            messagebox.showerror("Error", "No model loaded!")
            return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open webcam!")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            for r in results:
                frame = r.plot()
            cv2.imshow("Object Detection - Webcam", frame)  # No resizing in Webcam Mode
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.capture_screenshot(frame)  # Capture full-resolution frame
        cap.release()
        cv2.destroyAllWindows()

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection App")
        self.root.geometry("500x400")
        self.detector = None
        self.main_menu()

    def main_menu(self):
        self.clear_window()
        tk.Label(self.root, text="Main Menu", font=("Arial", 16, "bold")).pack(pady=20)

        tk.Button(self.root, text="Object Detection", width=25, height=2, command=self.object_detection_menu).pack(pady=10)
        tk.Button(self.root, text="Settings", width=25, height=2, command=self.settings_menu).pack(pady=10)
        tk.Button(self.root, text="View Screenshots", width=25, height=2, command=self.view_screenshots).pack(pady=10)
        tk.Button(self.root, text="Quit", width=25, height=2, bg="red", fg="white", command=self.root.quit).pack(pady=20)

        # Info button at bottom left
        info_button = tk.Button(self.root, text="ℹ Info", font=("Arial", 10, "bold"), bg="gray", fg="white", command=self.show_info)
        info_button.place(x=10, y=360)

    def object_detection_menu(self):
        self.clear_window()
        tk.Label(self.root, text="Object Detection", font=("Arial", 14, "bold")).pack(pady=10)

        tk.Button(self.root, text="Detect Image", width=25, command=self.detect_image).pack(pady=5)
        tk.Button(self.root, text="Detect Video", width=25, command=self.detect_video).pack(pady=5)
        tk.Button(self.root, text="Detect Webcam", width=25, command=self.detect_webcam).pack(pady=5)
        tk.Button(self.root, text="Back", width=25, bg="gray", command=self.main_menu).pack(pady=20)

    def settings_menu(self):
        self.clear_window()
        tk.Label(self.root, text="Settings", font=("Arial", 14, "bold")).pack(pady=10)

        tk.Button(self.root, text="Load Model", width=25, command=self.load_model).pack(pady=5)
        tk.Button(self.root, text="Gather Dataset", width=25, command=lambda: webbrowser.open("https://universe.roboflow.com/")).pack(pady=5)
        tk.Button(self.root, text="Train Model", width=25, command=lambda: webbrowser.open("https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb")).pack(pady=5)
        tk.Button(self.root, text="Back", width=25, bg="gray", command=self.main_menu).pack(pady=20)

    def load_model(self):
        file_path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=[("Model Files", "*.pt")])
        if file_path:
            self.detector = YOLODetector(file_path)
            messagebox.showinfo("Success", "Model loaded successfully!")

    def detect_image(self):
        if not self.detector:
            messagebox.showerror("Error", "Load a model first!")
            return
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            threading.Thread(target=self.detector.detect_image, args=(file_path,), daemon=True).start()

    def detect_video(self):
        if not self.detector:
            messagebox.showerror("Error", "Load a model first!")
            return
        file_path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if file_path:
            threading.Thread(target=self.detector.detect_video, args=(file_path,), daemon=True).start()

    def detect_webcam(self):
        if not self.detector:
            messagebox.showerror("Error", "Load a model first!")
            return
        threading.Thread(target=self.detector.detect_webcam, daemon=True).start()

    def view_screenshots(self):
        os.system("explorer screenshots")

    def show_info(self):
        info_text = (
            "📌 How to Use the App:\n"
            "1️⃣ Load a YOLO model in Settings.\n"
            "2️⃣ Use 'Detect Image', 'Detect Video', or 'Detect Webcam'.\n"
            "3️⃣ Press 's' during video/webcam to take a screenshot.\n"
            "4️⃣ View screenshots in 'View Screenshots'.\n"
            "5️⃣ Press 'q' to quit detection mode."
        )
        messagebox.showinfo("Instructions", info_text)

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
