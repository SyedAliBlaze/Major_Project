import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, ttk
import cv2
import numpy as np
import os
import threading
import webbrowser
from ultralytics import YOLO
import logging
from datetime import datetime
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import time
import sys
from queue import Queue
from time import perf_counter
import smtplib
from email.message import EmailMessage
import mimetypes
import math
import pkg_resources
import subprocess
import platform
import urllib.request
import tempfile
import json
from ttkthemes import ThemedTk

# Directory setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for folder in ["TEMP", "TEMP_RES", "RES", "MODEL", "folder result", "detections"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Settings file path
SETTINGS_FILE = os.path.join(SCRIPT_DIR, "settings.json")

# Default models and Gmail credentials
DEFAULT_YOLO_MODEL = os.path.join(SCRIPT_DIR, "MODEL", "yolo11s.pt")
DEFAULT_SR_MODEL = os.path.join(SCRIPT_DIR, "MODEL", "RealESRGAN_x4plus.pth")
GMAIL_USER = "projectblaze007@gmail.com"
GMAIL_PASSWORD = "dzry siey cjko anwv"  # Replace with your actual App Password

# Default settings
DEFAULT_SETTINGS = {
    "yolo_model": DEFAULT_YOLO_MODEL,
    "sr_model": DEFAULT_SR_MODEL,
    "object_list": ["person"],
    "threshold": 0.9,
    "gmail_user": GMAIL_USER,
    "gmail_password": GMAIL_PASSWORD,
    "receiver_email": "blazingsyedali451@gmail.com"
}

# Load settings from file or use defaults
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading settings: {e}. Using default settings.")
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()

# Save settings to file
def save_settings_to_file(settings):
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        print("Settings saved to file.")
    except IOError as e:
        print(f"Error saving settings: {e}")

SETTINGS = load_settings()

def check_python_version(required_version="3.9"):
    current_version = platform.python_version()
    if current_version.startswith(required_version):
        print(f"Python version {current_version} is compatible with the required version {required_version}.x.")
        return True
    else:
        print(f"Current Python version {current_version} does not match required version {required_version}.x.")
        return False

def find_python_39():
    try:
        result = subprocess.run(["py", "-3.9", "--version"], capture_output=True, text=True, check=False)
        if result.returncode == 0 and "Python 3.9" in result.stdout:
            print("Python 3.9 is installed on the system.")
            result = subprocess.run(["py", "-3.9", "-c", "import sys; print(sys.executable)"], 
                                    capture_output=True, text=True, check=True)
            return result.stdout.strip()
        else:
            print("Python 3.9 is not found among installed versions.")
            return None
    except subprocess.CalledProcessError:
        print("Error checking for Python 3.9 with 'py' launcher.")
        return None
    except FileNotFoundError:
        print("'py' launcher not found. Python Launcher may not be installed.")
        return None

def download_and_install_python_39():
    python_39_url = "https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe"
    installer_path = os.path.join(tempfile.gettempdir(), "python-3.9.13-amd64.exe")
    try:
        print(f"Downloading Python 3.9 installer from {python_39_url}...")
        urllib.request.urlretrieve(python_39_url, installer_path)
        print(f"Downloaded installer to {installer_path}")
        install_command = [installer_path, "/quiet", "InstallAllUsers=1", "PrependPath=1"]
        print("Installing Python 3.9. This may require Administrator privileges...")
        subprocess.run(install_command, check=True)
        print("Python 3.9 installation completed.")
        os.remove(installer_path)
        print(f"Cleaned up installer file: {installer_path}")
        new_python_path = find_python_39()
        if new_python_path:
            return new_python_path
        else:
            raise Exception("Python 3.9 installed but could not be located.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install Python 3.9: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during Python 3.9 installation: {e}")
        sys.exit(1)

def switch_to_python_39():
    if check_python_version("3.9"):
        return sys.executable
    python_39_path = find_python_39()
    if python_39_path:
        print(f"Found Python 3.9 at {python_39_path}. Relaunching script...")
        subprocess.run([python_39_path, *sys.argv], check=True)
        sys.exit(0)
    else:
        print("Python 3.9 not found. Proceeding to install...")
        python_39_path = download_and_install_python_39()
        print(f"Relaunching script with newly installed Python 3.9 at {python_39_path}...")
        subprocess.run([python_39_path, *sys.argv], check=True)
        sys.exit(0)

def check_and_install_requirements(requirements_file="requirements.txt"):
    try:
        with open(requirements_file, "r") as f:
            required = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        for req in required:
            try:
                pkg_name, pkg_version = req.split("==")
                pkg_name = pkg_name.lower()
                if pkg_name not in installed:
                    print(f"{pkg_name} is not installed. Installing {req}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", req])
                else:
                    installed_version = installed[pkg_name]
                    if installed_version != pkg_version:
                        print(f"{pkg_name} version mismatch: required {pkg_version}, installed {installed_version}. Reinstalling {req}...")
                        subprocess.check_call([sys.executable, "-m", "pip", "install", req, "--force-reinstall"])
                    else:
                        print(f"{pkg_name}=={pkg_version} is already installed and matches the required version.")
            except ValueError:
                print(f"Invalid requirement format: {req}. Skipping...")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {req}: {e}")
                sys.exit(1)
        print("All required packages are installed with the correct versions.")
    except FileNotFoundError:
        print(f"Error: {requirements_file} not found in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def suppress_yolo_logs():
    logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

def restore_logs():
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

def classify_resolution(width, height):
    total_pixels = width * height
    low_threshold = 500_000
    high_threshold = 2_000_000
    if total_pixels < low_threshold:
        return "Low Resolution"
    elif total_pixels >= high_threshold:
        return "High Resolution"
    else:
        return "Medium Resolution"

def get_available_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        except Exception as e:
            print(f"Camera {i} unavailable: {e}")
    return available_cameras

def get_contrasting_color(background_color):
    luminance = (0.299 * background_color[2] + 0.587 * background_color[1] + 0.114 * background_color[0]) / 255
    return (0, 0, 0) if luminance > 0.5 else (255, 255, 255)

def create_no_input_frame(width, height):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, "No Input Available", (width // 4, height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

class YOLODetector:
    def __init__(self, root=None):
        self.root = root
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.load_models()
        self.split_frames = 0
        self.processed_frames = 0
        self.stitched_frames = 0
        self.total_frames = 0
        self.cancelled = False

    def load_models(self):
        suppress_yolo_logs()
        try:
            self.yolo_model = YOLO(SETTINGS["yolo_model"]).to(self.device)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
            self.yolo_model = None
        try:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.sr_model = RealESRGANer(
                scale=4, model_path=SETTINGS["sr_model"], model=model, tile=0, tile_pad=10, pre_pad=0, device=self.device
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load SR model: {e}")
            self.sr_model = None
        restore_logs()

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
        if not self.yolo_model or not self.sr_model:
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
                results = self.yolo_model(img_resized)
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
        if not self.yolo_model:
            messagebox.showerror("Error", "No YOLO model loaded!")
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
                results = self.yolo_model(frame_resized)
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
        self.cancelled = False
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

    def save_detection_frame(self, frame):
        detection_dir = os.path.join(SCRIPT_DIR, 'detections')
        os.makedirs(detection_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(detection_dir, f'detection_{timestamp}.png')
        cv2.imwrite(filename, frame)
        return filename

    def email_alert_background(self, subject, body, frame):
        def send_email():
            try:
                detection_path = self.save_detection_frame(frame)
                msg = EmailMessage()
                msg.set_content(body)
                msg['Subject'] = subject
                msg['From'] = SETTINGS["gmail_user"]
                msg['To'] = SETTINGS["receiver_email"]
                
                with open(detection_path, 'rb') as img:
                    ctype, encoding = mimetypes.guess_type(detection_path)
                    if ctype is None or encoding is not None:
                        ctype = 'application/octet-stream'
                    maintype, subtype = ctype.split('/', 1)
                    msg.add_attachment(img.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(detection_path))
                
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(SETTINGS["gmail_user"], SETTINGS["gmail_password"])
                server.send_message(msg)
                server.quit()
                os.remove(detection_path)
                print("Email sent successfully")
            except smtplib.SMTPAuthenticationError:
                print("SMTP Authentication Error: Check your Gmail credentials or App Password")
            except smtplib.SMTPException as e:
                print(f"SMTP Error: {e}")
            except Exception as e:
                print(f"Error sending email: {e}")

        email_thread = threading.Thread(target=send_email)
        email_thread.daemon = True
        email_thread.start()

    def show_cameras(self, camera_indices, single_camera_index=None):
        if not self.yolo_model:
            messagebox.showerror("Error", "YOLO model not loaded!")
            return

        suppress_yolo_logs()
        caps = [cv2.VideoCapture(i) for i in camera_indices]
        num_cameras = len(camera_indices)
        grid_size = math.ceil(math.sqrt(num_cameras))
        screen_width, screen_height = 1400, 700
        cv2.namedWindow('Cameras', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Cameras', screen_width, screen_height)

        alert_queue = Queue()
        alert_thread = threading.Thread(target=lambda: self.show_alert(alert_queue))
        alert_thread.daemon = True
        alert_thread.start()

        selected_camera = single_camera_index if single_camera_index is not None else num_cameras - 1
        detection_start_times = {}
        email_sent = {}
        last_frame_time = perf_counter()

        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_camera
            if event == cv2.EVENT_LBUTTONDOWN:
                if selected_camera is not None:
                    if screen_width - 50 <= x <= screen_width - 10 and 10 <= y <= 50:
                        selected_camera = None
                    elif 10 <= x <= 50 and screen_height - 50 <= y <= screen_height - 10:
                        selected_camera = (selected_camera - 1) % num_cameras
                    elif screen_width - 50 <= x <= screen_width - 10 and screen_height - 50 <= y <= screen_height - 10:
                        selected_camera = (selected_camera + 1) % num_cameras
                else:
                    row = y // (frame_height + 1)
                    col = x // (frame_width + 1)
                    idx = row * grid_size + col
                    if idx < num_cameras:
                        selected_camera = idx

        cv2.setMouseCallback('Cameras', mouse_callback)

        while True:
            frames = []
            current_time = perf_counter()
            delta_time = current_time - last_frame_time
            last_frame_time = current_time

            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if ret:
                    try:
                        results = self.yolo_model(frame, stream=True)
                        frame_detections = set()
                        for r in results:
                            frame = r.plot()
                            for box in r.boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                if conf > SETTINGS["threshold"]:
                                    name = r.names[cls]
                                    frame_detections.add(name)
                                    if name in SETTINGS["object_list"]:
                                        key = (i, name)
                                        if key not in detection_start_times:
                                            detection_start_times[key] = current_time
                                            print(f"Started timer for {name} on Camera {i + 1} at {current_time:.2f}")
                                        duration = current_time - detection_start_times[key]
                                        print(f"Object {name} on Camera {i + 1} detected for {duration:.2f} seconds")
                                        if duration >= 4 and key not in email_sent:
                                            self.email_alert_background(
                                                f"Object {name} Detected",
                                                f"Object '{name}' has been detected for more than 4 seconds in Camera {i + 1}.",
                                                frame
                                            )
                                            email_sent[key] = True
                                            alert_queue.put(name)
                                        cv2.putText(frame, f"ALERT: {name} ({conf:.2f})", (10, 30),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 3)
                        for key in list(detection_start_times.keys()):
                            cam_idx, obj_name = key
                            if cam_idx == i and obj_name not in frame_detections:
                                print(f"Resetting timer for {obj_name} on Camera {cam_idx + 1}")
                                del detection_start_times[key]
                                if key in email_sent:
                                    del email_sent[key]
                    except Exception as e:
                        print(f"Error in object detection: {e}")
                else:
                    frame = create_no_input_frame(640, 480)
                frames.append(frame)

            screen_width = cv2.getWindowImageRect('Cameras')[2]
            screen_height = cv2.getWindowImageRect('Cameras')[3]
            frame_height = screen_height // grid_size
            frame_width = screen_width // grid_size
            resized_frames = [cv2.resize(frame, (frame_width, frame_height)) for frame in frames]

            while len(resized_frames) < grid_size * grid_size:
                resized_frames.append(create_no_input_frame(frame_width, frame_height))

            if selected_camera is not None:
                enlarged_frame = cv2.resize(frames[selected_camera], (screen_width, screen_height))
                cv2.rectangle(enlarged_frame, (screen_width - 50, 10), (screen_width - 10, 50), (0, 0, 255), -1)
                cv2.putText(enlarged_frame, "X", (screen_width - 40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.rectangle(enlarged_frame, (10, screen_height - 50), (50, screen_height - 10), (0, 255, 0), -1)
                cv2.putText(enlarged_frame, "<", (15, screen_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.rectangle(enlarged_frame, (screen_width - 50, screen_height - 50), (screen_width - 10, screen_height - 10), (0, 255, 0), -1)
                cv2.putText(enlarged_frame, ">", (screen_width - 40, screen_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                font_color = get_contrasting_color(enlarged_frame[0, 0])
                cv2.putText(enlarged_frame, f"Camera {selected_camera + 1}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
                cv2.imshow('Cameras', enlarged_frame)
            else:
                grid_frame = np.zeros((frame_height * grid_size + 1 * (grid_size - 1),
                                      frame_width * grid_size + 1 * (grid_size - 1), 3), dtype=np.uint8)
                grid_frame.fill(255)
                for idx, frame in enumerate(resized_frames):
                    row, col = divmod(idx, grid_size)
                    y1 = row * (frame_height + 1)
                    x1 = col * (frame_width + 1)
                    grid_frame[y1:y1 + frame_height, x1:x1 + frame_width] = frame
                    if idx < num_cameras:
                        font_color = get_contrasting_color(frame[0, 0])
                        cv2.putText(grid_frame, f"Camera {idx + 1}", (x1 + 10, y1 + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
                cv2.imshow('Cameras', grid_frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Cameras', cv2.WND_PROP_VISIBLE) < 1:
                break

        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
        restore_logs()
        alert_queue.put(None)
        alert_thread.join()

    def show_alert(self, alert_queue):
        alert_cooldown = {}
        cooldown_period = 5
        while True:
            try:
                obj = alert_queue.get(timeout=0.1)
                if obj is None:
                    break
                current_time = perf_counter()
                if current_time - alert_cooldown.get(obj, 0) >= cooldown_period:
                    messagebox.showinfo("Alert", f"Object '{obj}' detected!")
                    alert_cooldown[obj] = current_time
            except Queue.Empty:
                continue
            except Exception as e:
                print(f"Error in alert thread: {e}")

class CombinedDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Object Detection System")
        self.root.geometry("600x500")
        self.root.configure(bg="#0a0a0a")
        self.style = ttk.Style()
        self.apply_theme()
        
        self.detector = YOLODetector(root=self.root)
        self.model_name = tk.StringVar()
        self.model_name.set(f"YOLO: {os.path.basename(SETTINGS['yolo_model'])}, SR: {os.path.basename(SETTINGS['sr_model'])}")
        self.progress_window = None
        self.split_label = None
        self.processed_label = None
        self.stitched_label = None
        self.active_threads = []
        self.main_menu()

    def apply_theme(self):
        """Apply a futuristic style to ttk widgets"""
        self.style.theme_use("clam")
        self.style.configure("TButton", font=("Orbitron", 12), padding=10, 
                           background="#000", foreground="#00ffff", borderwidth=0)
        self.style.map("TButton", background=[("active", "#005f5f")])
        self.style.configure("TLabel", font=("Orbitron", 14), 
                           background="#0a0a0a", foreground="#00ffff")

    def create_futuristic_button(self, text, command, color="#00ffff", width=25):
        button = ttk.Button(self.root, text=text, command=command, 
                          style="TButton", width=width)
        button.pack(pady=10, ipadx=10, ipady=5, fill="x")
        return button

    def main_menu(self):
        self.clear_window()
        ttk.Label(self.root, text="Advanced Object Detection", 
                 font=("Orbitron", 20)).pack(pady=20)
        self.create_futuristic_button("Start Detection", self.detection_options_menu)
        self.create_futuristic_button("Customize Model", self.customize_model_menu)
        self.create_futuristic_button("Settings", self.settings_menu)
        self.create_futuristic_button("View Results", self.view_results_menu)
        self.create_futuristic_button("Quit", self.quit_app, color="#ff0033")
        info_button = ttk.Button(self.root, text="ℹ Info", 
                               command=self.show_info,
                               style="TButton")
        info_button.place(x=10, y=460)

    def detection_options_menu(self):
        self.clear_window()
        ttk.Label(self.root, text="Detection Options").pack(pady=10)
        self.create_futuristic_button("Detect Folder", self.detect_folder)
        self.create_futuristic_button("Process Video", self.process_video)
        self.create_futuristic_button("Camera Detection", self.start_camera_detection)
        self.create_futuristic_button("Back", self.main_menu, color="#808080")

    def customize_model_menu(self):
        self.clear_window()
        ttk.Label(self.root, text="Customize Model").pack(pady=10)
        ttk.Label(self.root, textvariable=self.model_name, 
                 font=("Orbitron", 12)).pack(pady=5)
        self.create_futuristic_button("Load YOLO Model", self.load_yolo_model)
        self.create_futuristic_button("Load SR Model", self.load_sr_model)
        self.create_futuristic_button("Gather Dataset", 
                                    lambda: webbrowser.open("https://universe.roboflow.com/"))
        self.create_futuristic_button("Train Model", 
                                    lambda: webbrowser.open("https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb"))
        self.create_futuristic_button("Source Code", self.open_source_code_dir)
        self.create_futuristic_button("Back", self.main_menu, color="#808080")

    def open_source_code_dir(self):
        script_dir = os.path.abspath(SCRIPT_DIR)
        try:
            if os.name == 'nt':  # Windows
                os.startfile(script_dir)
            elif os.name == 'posix':  # macOS/Linux
                opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
                subprocess.Popen([opener, script_dir])
            else:
                messagebox.showerror("Error", "Unsupported operating system.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open directory: {e}")

    def detect_folder(self):
        folder_path = filedialog.askdirectory(title="Select a Folder")
        if folder_path:
            self.show_loading_popup(folder_path)

    def show_loading_popup(self, folder_path):
        loading_popup = Toplevel(self.root)
        loading_popup.title("Processing")
        loading_popup.geometry("300x200")
        loading_popup.configure(bg="#0a0a0a")
        ttk.Label(loading_popup, text="Processing images...").pack(pady=10)
        status_label = ttk.Label(loading_popup, text="Processed 0/0 images")
        status_label.pack(pady=10)
        timer_label = ttk.Label(loading_popup, text="Elapsed Time: 0s")
        timer_label.pack(pady=10)
        resolution_label = ttk.Label(loading_popup, text="Low Res: 0, Med/High Res: 0")
        resolution_label.pack(pady=10)
        def run_detection():
            try:
                self.detector.detect_folder(folder_path, status_label, timer_label, resolution_label)
            except Exception as e:
                self.root.after(0, lambda: status_label.config(text=f"Error: {e}"))
        thread = threading.Thread(target=run_detection)
        thread.daemon = True
        self.active_threads.append(thread)
        thread.start()

    def process_video(self):
        file_path = filedialog.askopenfilename(title="Select a Video", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.create_progress_window()
            thread = threading.Thread(target=self.detector.process_video, args=(file_path, self))
            thread.daemon = True
            self.active_threads.append(thread)
            thread.start()

    def create_progress_window(self):
        self.progress_window = Toplevel(self.root)
        self.progress_window.title("Processing Progress")
        self.progress_window.geometry("300x250")
        self.progress_window.configure(bg="#0a0a0a")
        self.progress_window.transient(self.root)
        self.progress_window.grab_set()
        ttk.Label(self.progress_window, text="Processing Video...", 
                 font=("Orbitron", 12)).pack(pady=10)
        self.split_label = ttk.Label(self.progress_window, 
                                   text="Frames Split: 0 / 0")
        self.split_label.pack(pady=5)
        self.processed_label = ttk.Label(self.progress_window, 
                                       text="Frames Processed: 0 / 0")
        self.processed_label.pack(pady=5)
        self.stitched_label = ttk.Label(self.progress_window, 
                                      text="Frames Stitched: 0 / 0")
        self.stitched_label.pack(pady=5)
        self.create_futuristic_button("Cancel", self.cancel_video_processing, 
                                    color="#ff0033", width=15)
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

    def start_camera_detection(self):
        self.root.withdraw()
        available_cameras = get_available_cameras()
        if not available_cameras:
            print("No cameras found.")
            self.root.deiconify()
            messagebox.showerror("Error", "No cameras found.")
            return
        print(f"Available cameras: {available_cameras}")
        last_camera_index = available_cameras[-1]
        self.detector.show_cameras(available_cameras, single_camera_index=last_camera_index)
        self.root.deiconify()

    def view_results_menu(self):
        self.clear_window()
        ttk.Label(self.root, text="View Results").pack(pady=10)
        self.create_futuristic_button("View Folder Results", self.view_folder_results)
        self.create_futuristic_button("View Video Results", self.view_video_results)
        self.create_futuristic_button("View Detections", self.view_detections)
        self.create_futuristic_button("Back", self.main_menu, color="#808080")

    def view_folder_results(self):
        folder_result_path = os.path.abspath("folder result")
        if os.path.exists(folder_result_path):
            os.startfile(folder_result_path)
        else:
            messagebox.showinfo("Info", "The 'folder result' directory does not exist yet.")

    def view_video_results(self):
        os.startfile(os.path.abspath("RES"))

    def view_detections(self):
        detections_path = os.path.abspath("detections")
        if os.path.exists(detections_path):
            os.startfile(detections_path)
        else:
            messagebox.showinfo("Info", "The 'detections' directory does not exist yet or is empty.")

    def show_info(self):
        info_text = (
            "📌 Advanced Object Detection System\n\n"
            "1️⃣ **Detection Options**: Choose detection mode.\n"
            "   - Detect Folder: Process images, results in 'folder result'.\n"
            "   - Process Video: Process videos, results in 'RES'.\n"
            "   - Camera Detection: Real-time monitoring, frames in 'detections'.\n\n"
            "2️⃣ **Customize Model**: Load models and access training resources.\n\n"
            "3️⃣ **Settings**: Adjust detection parameters.\n\n"
            "4️⃣ **View Results**: Access processed outputs.\n\n"
            "⚠ Ensure models are in 'MODEL' folder.\n"
            "Developed by **Syed Ali N.** 🚀"
        )
        messagebox.showinfo("Application Info", info_text)

    def settings_menu(self):
        self.clear_window()
        ttk.Label(self.root, text="Settings").pack(pady=10)
        
        ttk.Label(self.root, text="Gmail Address:").pack(pady=5)
        self.gmail_entry = tk.Entry(self.root, width=30, bg="#1a1a1a", 
                                  fg="#00ffff", insertbackground="#00ffff")
        self.gmail_entry.insert(0, SETTINGS["gmail_user"])
        self.gmail_entry.pack(pady=5)
        
        ttk.Label(self.root, text="Gmail App Password:").pack(pady=5)
        self.password_entry = tk.Entry(self.root, width=30, show="*", 
                                     bg="#1a1a1a", fg="#00ffff", 
                                     insertbackground="#00ffff")
        self.password_entry.insert(0, SETTINGS["gmail_password"])
        self.password_entry.pack(pady=5)
        
        ttk.Label(self.root, text="Receiver Email:").pack(pady=5)
        self.receiver_entry = tk.Entry(self.root, width=30, bg="#1a1a1a", 
                                     fg="#00ffff", insertbackground="#00ffff")
        self.receiver_entry.insert(0, SETTINGS["receiver_email"])
        self.receiver_entry.pack(pady=5)
        
        ttk.Label(self.root, text="Objects to Detect (comma-separated):").pack(pady=5)
        self.object_entry = tk.Entry(self.root, width=30, bg="#1a1a1a", 
                                   fg="#00ffff", insertbackground="#00ffff")
        self.object_entry.insert(0, ", ".join(SETTINGS["object_list"]))
        self.object_entry.pack(pady=5)
        
        ttk.Label(self.root, text="Confidence Threshold (0.0-1.0):").pack(pady=5)
        self.threshold_entry = tk.Entry(self.root, width=10, bg="#1a1a1a", 
                                      fg="#00ffff", insertbackground="#00ffff")
        self.threshold_entry.insert(0, str(SETTINGS["threshold"]))
        self.threshold_entry.pack(pady=5)
        
        self.create_futuristic_button("Save Settings", self.save_settings)
        self.create_futuristic_button("Back", self.main_menu, color="#808080")

    def load_yolo_model(self):
        file_path = filedialog.askopenfilename(initialdir=os.path.join(SCRIPT_DIR, "MODEL"),
                                               title="Select YOLO Model", filetypes=[("Model Files", "*.pt")])
        if file_path:
            SETTINGS["yolo_model"] = file_path
            self.detector.load_models()
            self.model_name.set(f"YOLO: {os.path.basename(SETTINGS['yolo_model'])}, SR: {os.path.basename(SETTINGS['sr_model'])}")
            save_settings_to_file(SETTINGS)
            messagebox.showinfo("Success", "YOLO model loaded successfully!")

    def load_sr_model(self):
        file_path = filedialog.askopenfilename(initialdir=os.path.join(SCRIPT_DIR, "MODEL"),
                                               title="Select SR Model", filetypes=[("Model Files", "*.pth")])
        if file_path:
            SETTINGS["sr_model"] = file_path
            self.detector.load_models()
            self.model_name.set(f"YOLO: {os.path.basename(SETTINGS['yolo_model'])}, SR: {os.path.basename(SETTINGS['sr_model'])}")
            save_settings_to_file(SETTINGS)
            messagebox.showinfo("Success", "SR model loaded successfully!")

    def save_settings(self):
        gmail_user = self.gmail_entry.get().strip()
        gmail_password = self.password_entry.get().strip()
        receiver_email = self.receiver_entry.get().strip()
        if not gmail_user or not gmail_password or not receiver_email:
            messagebox.showerror("Error", "Gmail address, password, and receiver email cannot be empty!")
            return
        
        SETTINGS["gmail_user"] = gmail_user
        SETTINGS["gmail_password"] = gmail_password
        SETTINGS["receiver_email"] = receiver_email
        SETTINGS["object_list"] = [obj.strip() for obj in self.object_entry.get().split(',')]
        
        try:
            SETTINGS["threshold"] = float(self.threshold_entry.get())
            if not 0 <= SETTINGS["threshold"] <= 1:
                raise ValueError("Threshold must be between 0 and 1")
            if not os.path.exists(SETTINGS["yolo_model"]):
                raise FileNotFoundError(f"YOLO model file '{SETTINGS['yolo_model']}' not found")
            if not os.path.exists(SETTINGS["sr_model"]):
                raise FileNotFoundError(f"SR model file '{SETTINGS['sr_model']}' not found")
            save_settings_to_file(SETTINGS)
            messagebox.showinfo("Settings", "Settings saved successfully!")
            self.main_menu()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid threshold value: {e}")
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))

    def quit_app(self):
        self.detector.cancel_processing()
        if self.progress_window and self.progress_window.winfo_exists():
            self.progress_window.destroy()
        self.root.destroy()
        sys.exit(0)

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    python_39_executable = switch_to_python_39()
    check_and_install_requirements("requirements.txt")
    root = ThemedTk(theme="equilux")
    app = CombinedDetectionApp(root)
    root.mainloop()