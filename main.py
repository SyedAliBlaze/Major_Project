import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
import cv2
import os
import threading
import webbrowser
from ultralytics import YOLO
import logging
from datetime import datetime
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet  # Real-ESRGAN dependency
from realesrgan import RealESRGANer  # Real-ESRGAN dependency
import time  # For timer functionality
import sys  # For proper exit

# Create directories
for folder in ["TEMP", "TEMP_RES", "RES", "MODEL", "folder result"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

DEFAULT_MODEL = os.path.join("MODEL", "yolo11s.pt")
DEFAULT_SR_MODEL = os.path.join("MODEL", "RealESRGAN_x4plus.pth")  # Super-resolution model

def suppress_yolo_logs():
    logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

def restore_logs():
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

def classify_resolution(width, height):
    total_pixels = width * height
    low_threshold = 500_000  # 0.5 million pixels
    high_threshold = 2_000_000  # 2 megapixels
    if total_pixels < low_threshold:
        return "Low Resolution"
    elif total_pixels >= high_threshold:
        return "High Resolution"
    else:
        return "Medium Resolution"

class YOLODetector:
    def __init__(self, model_path=DEFAULT_MODEL, sr_model_path=DEFAULT_SR_MODEL, root=None):
        self.root = root  # For thread-safe GUI updates
        self.model_path = model_path
        self.sr_model_path = sr_model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        suppress_yolo_logs()
        self.model = self.load_model(model_path)
        self.sr_model = self.load_sr_model(sr_model_path)
        restore_logs()
        self.split_frames = 0
        self.processed_frames = 0
        self.stitched_frames = 0
        self.total_frames = 0
        self.cancelled = False

    def load_model(self, model_path):
        try:
            if not os.path.isabs(model_path):
                model_path = os.path.join(os.path.dirname(__file__), model_path)
            model = YOLO(model_path)
            model.to(self.device)
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
            return None

    def load_sr_model(self, sr_model_path):
        try:
            if not os.path.isabs(sr_model_path):
                sr_model_path = os.path.join(os.path.dirname(__file__), sr_model_path)
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=4, model_path=sr_model_path, model=model, tile=0, tile_pad=10, pre_pad=0, device=self.device
            )
            return upsampler
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load SR model: {e}")
            return None

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

    def apply_super_resolution(self, img):
        if not self.sr_model:
            return img
        try:
            sr_img, _ = self.sr_model.enhance(img, outscale=4)
            return sr_img
        except Exception as e:
            messagebox.showerror("Error", f"Super-resolution failed: {e}")
            return img

    def capture_screenshot(self, frame, output_path):
        cv2.imwrite(output_path, frame)

    def detect_folder(self, folder_path, status_label, timer_label, resolution_label):
        if not self.model or not self.sr_model:
            messagebox.showerror("Error", "Model(s) not loaded!")
            return

        date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_subfolder = os.path.join("folder result", date_time_str)
        os.makedirs(result_subfolder, exist_ok=True)

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_files = len(image_files)
        processed_files = 0
        low_res_count = 0
        high_med_res_count = 0

        if total_files == 0:
            messagebox.showwarning("Warning", "No valid images found!")
            self.root.after(0, lambda: status_label.config(text="No valid images found!"))
            return

        start_time = time.time()
        for filename in image_files:
            file_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(file_path)
                if img is None:
                    continue
                height, width = img.shape[:2]
                resolution_class = classify_resolution(width, height)
                if resolution_class == "Low Resolution":
                    low_res_count += 1
                    img_processed = self.apply_super_resolution(img)
                else:
                    high_med_res_count += 1
                    img_processed = img
                img_resized = self.resize_frame(img_processed)
                results = self.model(img_resized)
                img_with_boxes = img_resized.copy()
                for r in results:
                    img_with_boxes = r.plot()
                output_path = os.path.join(result_subfolder, filename)
                self.capture_screenshot(img_with_boxes, output_path)
                processed_files += 1
                elapsed_time = int(time.time() - start_time)
                self.root.after(0, lambda: status_label.config(text=f"Processed {processed_files}/{total_files} images"))
                self.root.after(0, lambda: timer_label.config(text=f"Elapsed Time: {elapsed_time}s"))
            except Exception:
                continue

        elapsed_time = int(time.time() - start_time)
        self.root.after(0, lambda: status_label.config(text="Processing complete!"))
        self.root.after(0, lambda: timer_label.config(text=f"Elapsed Time: {elapsed_time}s"))
        self.root.after(0, lambda: resolution_label.config(text=f"Low Res: {low_res_count}, Med/High Res: {high_med_res_count}"))
        self.root.latest_result_subfolder = result_subfolder

    def split_video_to_frames(self, video_path):
        self.split_frames = 0
        self.total_frames = 0
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file!")
            return None, None
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while cap.isOpened() and not self.cancelled:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join("TEMP", f"frame_{self.split_frames:06d}.png")
            cv2.imwrite(frame_path, frame)
            self.split_frames += 1
        cap.release()
        if self.cancelled:
            return None, None
        return fps, self.split_frames

    def process_frames_with_yolo(self):
        if not self.model:
            messagebox.showerror("Error", "No model loaded!")
            return
        self.processed_frames = 0
        for filename in os.listdir("TEMP"):
            if self.cancelled:
                break
            if filename.endswith(".png"):
                frame_path = os.path.join("TEMP", filename)
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                height, width = frame.shape[:2]
                resolution_class = classify_resolution(width, height)
                if resolution_class == "Low Resolution":
                    frame = self.apply_super_resolution(frame)
                frame_resized = self.resize_frame(frame)
                results = self.model(frame_resized)
                for r in results:
                    frame_resized = r.plot()
                output_path = os.path.join("TEMP_RES", filename)
                cv2.imwrite(output_path, frame_resized)
                self.processed_frames += 1

    def stitch_frames_to_video(self, fps, output_video_path):
        self.stitched_frames = 0
        images = [img for img in os.listdir("TEMP_RES") if img.endswith(".png")]
        images.sort()
        if not images:
            messagebox.showerror("Error", "No processed frames found!")
            return
        frame = cv2.imread(os.path.join("TEMP_RES", images[0]))
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        for image in images:
            if self.cancelled:
                break
            frame = cv2.imread(os.path.join("TEMP_RES", image))
            out.write(frame)
            self.stitched_frames += 1
        out.release()

    def cleanup_temp_files(self):
        for folder in ["TEMP", "TEMP_RES"]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception:
                    pass

    def process_video(self, video_path, app):
        self.cancelled = False  # Reset cancellation flag
        fps, frame_count = self.split_video_to_frames(video_path)
        if fps is None or frame_count == 0:
            if not self.cancelled:
                messagebox.showerror("Error", "Video splitting failed!")
            app.progress_window.destroy()
            return
        self.process_frames_with_yolo()
        if not self.cancelled:
            output_video_path = os.path.join("RES", f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            self.stitch_frames_to_video(fps, output_video_path)
        self.cleanup_temp_files()
        app.progress_window.destroy()
        if self.cancelled:
            messagebox.showinfo("Cancelled", "Video processing was cancelled.")

    def cancel_processing(self):
        self.cancelled = True

class CombinedDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection with SR (Folder & Video)")
        self.root.geometry("500x400")
        self.detector = YOLODetector(root=self.root)
        self.model_name = tk.StringVar()
        self.model_name.set(f"Model: {os.path.basename(DEFAULT_MODEL)}, SR: {os.path.basename(DEFAULT_SR_MODEL)}")
        self.progress_window = None
        self.split_label = None
        self.processed_label = None
        self.stitched_label = None
        self.active_threads = []  # Track active threads
        self.main_menu()

    def main_menu(self):
        self.clear_window()
        tk.Label(self.root, text="Object Detection with SR", font=("Arial", 16, "bold")).pack(pady=20)
        tk.Button(self.root, text="Detect Folder", width=25, height=2, command=self.detect_folder).pack(pady=10)
        tk.Button(self.root, text="Process Video", width=25, height=2, command=self.process_video).pack(pady=10)
        tk.Button(self.root, text="Settings", width=25, height=2, command=self.settings_menu).pack(pady=10)
        tk.Button(self.root, text="View Results", width=25, height=2, command=self.view_results_menu).pack(pady=10)
        tk.Button(self.root, text="Quit", width=25, height=2, bg="red", fg="white", command=self.quit_app).pack(pady=20)
        info_button = tk.Button(self.root, text="ℹ Info", font=("Arial", 10, "bold"), bg="gray", fg="white", command=self.show_info)
        info_button.place(x=10, y=360)

    def detect_folder(self):
        folder_path = filedialog.askdirectory(title="Select a Folder")
        if folder_path:
            self.show_loading_popup(folder_path)

    def show_loading_popup(self, folder_path):
        loading_popup = Toplevel(self.root)
        loading_popup.title("Processing")
        loading_popup.geometry("300x200")
        tk.Label(loading_popup, text="Processing images...").pack(pady=10)
        status_label = tk.Label(loading_popup, text="Processed 0/0 images")
        status_label.pack(pady=10)
        timer_label = tk.Label(loading_popup, text="Elapsed Time: 0s")
        timer_label.pack(pady=10)
        resolution_label = tk.Label(loading_popup, text="Low Res: 0, Med/High Res: 0")
        resolution_label.pack(pady=10)

        def run_detection():
            try:
                self.detector.detect_folder(folder_path, status_label, timer_label, resolution_label)
            except Exception as e:
                self.root.after(0, lambda: status_label.config(text=f"Error: {e}"))

        thread = threading.Thread(target=run_detection)
        thread.daemon = True  # Make thread daemon so it stops when the main program exits
        self.active_threads.append(thread)
        thread.start()

    def process_video(self):
        file_path = filedialog.askopenfilename(title="Select a Video", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.create_progress_window()
            thread = threading.Thread(target=self.detector.process_video, args=(file_path, self))
            thread.daemon = True  # Make thread daemon
            self.active_threads.append(thread)
            thread.start()

    def create_progress_window(self):
        self.progress_window = Toplevel(self.root)
        self.progress_window.title("Processing Progress")
        self.progress_window.geometry("300x250")
        self.progress_window.transient(self.root)
        self.progress_window.grab_set()
        tk.Label(self.progress_window, text="Processing Video...", font=("Arial", 12, "bold")).pack(pady=10)
        self.split_label = tk.Label(self.progress_window, text="Frames Split: 0 / 0", font=("Arial", 10))
        self.split_label.pack(pady=5)
        self.processed_label = tk.Label(self.progress_window, text="Frames Processed: 0 / 0", font=("Arial", 10))
        self.processed_label.pack(pady=5)
        self.stitched_label = tk.Label(self.progress_window, text="Frames Stitched: 0 / 0", font=("Arial", 10))
        self.stitched_label.pack(pady=5)
        tk.Button(self.progress_window, text="Cancel", width=15, bg="red", fg="white", command=self.cancel_video_processing).pack(pady=20)
        self.update_progress()

    def update_progress(self):
        if self.progress_window and self.progress_window.winfo_exists():
            self.split_label.config(text=f"Frames Split: {self.detector.split_frames} / {self.detector.total_frames}")
            self.processed_label.config(text=f"Frames Processed: {self.detector.processed_frames} / {self.detector.total_frames}")
            self.stitched_label.config(text=f"Frames Stitched: {self.detector.stitched_frames} / {self.detector.total_frames}")
            self.progress_window.after(100, self.update_progress)

    def cancel_video_processing(self):
        self.detector.cancel_processing()
        if self.progress_window and self.progress_window.winfo_exists():
            self.progress_window.destroy()

    def view_results_menu(self):
        self.clear_window()
        tk.Label(self.root, text="View Results", font=("Arial", 14, "bold")).pack(pady=10)
        tk.Button(self.root, text="View Folder Results", width=25, command=self.view_folder_results).pack(pady=10)
        tk.Button(self.root, text="View Video Results", width=25, command=self.view_video_results).pack(pady=10)
        tk.Button(self.root, text="Back", width=25, bg="gray", command=self.main_menu).pack(pady=20)

    def view_folder_results(self):
        folder_result_path = os.path.abspath("folder result")
        if os.path.exists(folder_result_path):
            os.startfile(folder_result_path)
        else:
            messagebox.showinfo("Info", "The 'folder result' directory does not exist yet. Process a folder to create it.")

    def view_video_results(self):
        os.startfile(os.path.abspath("RES"))

    def show_info(self):
        info_text = (
            "📌 How to Use Object Detection with Super-Resolution\n\n"
            "1️⃣ **Detect Folder**:\n"
            "   - Select a folder with images (.jpg, .png, .jpeg).\n"
            "   - Low-res images (<0.5MP) are upscaled then processed with YOLO.\n"
            "   - Results saved in 'folder result/YYYY-MM-DD_HH-MM-SS'.\n\n"
            "2️⃣ **Process Video**:\n"
            "   - Select a video (.mp4, .avi, .mov).\n"
            "   - Frames are split, low-res frames upscaled, processed with YOLO, then stitched back.\n"
            "   - Results saved in 'RES'. Use 'Cancel' to stop processing.\n\n"
            "3️⃣ **Settings**:\n"
            "   - Load a custom YOLO model (.pt) from 'MODEL' folder.\n\n"
            "4️⃣ **View Results**:\n"
            "   - Choose to view folder or video results.\n\n"
            "⚠ Note: Ensure YOLO and SR models are in 'MODEL'.\n"
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
        file_path = filedialog.askopenfilename(initialdir=os.path.join(os.path.dirname(__file__), "MODEL"),
                                               title="Select YOLO Model", filetypes=[("Model Files", "*.pt")])
        if file_path:
            self.detector = YOLODetector(model_path=file_path, sr_model_path=DEFAULT_SR_MODEL, root=self.root)
            self.model_name.set(f"Model: {os.path.basename(file_path)}, SR: {os.path.basename(DEFAULT_SR_MODEL)}")
            messagebox.showinfo("Success", "Model loaded successfully!")

    def quit_app(self):
        # Cancel any ongoing video processing
        self.detector.cancel_processing()
        # Close any open progress windows
        if self.progress_window and self.progress_window.winfo_exists():
            self.progress_window.destroy()
        # Destroy the main window and exit
        self.root.destroy()
        sys.exit(0)  # Ensure the program fully terminates

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CombinedDetectionApp(root)
    root.mainloop()