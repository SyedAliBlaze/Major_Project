import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, Toplevel
from PIL import Image, ImageTk
from ultralytics import YOLO
import time  # For unique filename timestamps

# Initialize Tkinter
root = tk.Tk()
root.title("YOLO Object Detection")
root.geometry("400x300")

# Load YOLO Model
model_path = "yolo11s.pt"
model = YOLO(model_path, task='detect')
labels = model.names

# Global Variables
cap = None
running = False
paused = False  # Pause flag
detection_window = None
panel = None
img_ref = None  # Prevent garbage collection
current_frame = None  # Stores the last displayed frame


def open_detection_window(width, height):
    """Opens a separate detection window."""
    global detection_window, panel, running

    root.withdraw()  # Hide main menu
    running = True

    detection_window = Toplevel(root)
    detection_window.title("Detection Window")
    detection_window.geometry(f"{width}x{height}")

    panel = tk.Label(detection_window)
    panel.pack()

    detection_window.protocol("WM_DELETE_WINDOW", close_detection_window)
    detection_window.bind("<KeyPress>", key_press_event)  # Capture key events
    detection_window.focus_set()  # Ensure key events are captured


def close_detection_window():
    """Closes the detection window and stops processing."""
    global running, cap, detection_window
    running = False
    if cap:
        cap.release()

    if detection_window:
        detection_window.destroy()
        detection_window = None

    root.deiconify()  # Show main menu again


def key_press_event(event):
    """Handles key events for pausing and taking screenshots."""
    global paused, current_frame

    if event.char == 'p':
        paused = not paused  # Toggle pause state
    elif event.char == 's' and current_frame is not None:
        save_screenshot()


def save_screenshot():
    """Saves the current frame as an image file in the 'screenshots' folder."""
    global current_frame
    if current_frame is None:
        return

    # Create the screenshots folder if it doesn't exist
    screenshot_folder = "screenshots"
    if not os.path.exists(screenshot_folder):
        os.makedirs(screenshot_folder)

    # Generate a unique filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(screenshot_folder, f"screenshot_{timestamp}.png")

    # Save the image
    cv2.imwrite(filename, cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR))
    print(f"Screenshot saved as {filename}")



def select_image():
    """Opens a file dialog for image selection."""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        image = cv2.imread(file_path)
        height, width, _ = image.shape

        open_detection_window(600, int((600 / width) * height))  # Dynamic height
        process_image(image)


def process_image(frame):
    """Processes an image by applying YOLO object detection."""
    frame = detect_objects(frame)

    # Convert BGR to RGB to fix discoloration
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    original_height, original_width = frame.shape[:2]
    new_width = 600
    new_height = int((new_width / original_width) * original_height)

    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    display_frame(frame)



def detect_objects(frame):
    """Runs YOLO object detection on the given frame."""
    results = model(frame, verbose=False)
    detections = results[0].boxes

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > 0.5:
            color = (0, 255, 0)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf * 100)}%'
            cv2.putText(frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return frame


def select_video():
    """Opens a file dialog for video selection."""
    global cap
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.avi;*.mp4;*.mkv")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Reduce resolution to half while maintaining aspect ratio
        new_width = original_width // 2
        new_height = original_height // 2

        open_detection_window(new_width, new_height)
        process_video(new_width, new_height)


def use_webcam():
    """Opens the webcam for real-time object detection."""
    global cap
    cap = cv2.VideoCapture(0)  # Open default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set webcam resolution to 1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    open_detection_window(1280, 720)
    process_video(1280, 720)


def process_video(new_width, new_height):
    """Processes video or webcam frames, handling pause and detection."""
    global cap, running, paused, panel, img_ref, current_frame

    if not running or not cap.isOpened():
        close_detection_window()
        return

    if paused:
        panel.after(10, lambda: process_video(new_width, new_height))  # Keep checking while paused
        return

    ret, frame = cap.read()
    if not ret:
        close_detection_window()
        return

    # Flip only if using webcam (assuming resolution 1280x720)
    if cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280 and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 720:
        frame = cv2.flip(frame, 1)  # Flip webcam feed

    # Resize the video frame to half resolution while keeping aspect ratio
    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    frame = detect_objects(frame)  # Apply YOLO detection

    current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Store current frame
    display_frame(current_frame)

    if running:
        panel.after(10, lambda: process_video(new_width, new_height))  # Refresh every 10ms


def display_frame(frame):
    """Displays the processed frame on the Tkinter panel."""
    global img_ref

    if not detection_window or not detection_window.winfo_exists():
        return

    img = Image.fromarray(frame)
    img_ref = ImageTk.PhotoImage(image=img)
    panel.config(image=img_ref)
    panel.update_idletasks()


# Tkinter UI Buttons
btn_image = tk.Button(root, text="Select Image", command=select_image)
btn_image.pack(pady=10)

btn_video = tk.Button(root, text="Select Video", command=select_video)
btn_video.pack(pady=10)

btn_webcam = tk.Button(root, text="Use Webcam", command=use_webcam)
btn_webcam.pack(pady=10)

root.mainloop()
