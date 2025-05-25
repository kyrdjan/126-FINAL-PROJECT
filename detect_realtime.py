import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import torch
from PIL import Image, ImageTk

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="runs/train/exp6/weights/best.pt", force_reload=True)

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Initialize GUI window
window = tk.Tk()
window.title("YOLOv5 Detection GUI")
window.geometry("800x600")

# Image display label (used for images and video frames)
image_label = tk.Label(window)
image_label.pack()

processed_video_path = None
video_capture = None
video_playing = False
play_width, play_height = 640, 480  # fixed playback size
progress_label = None

# Buttons Frame for side-by-side layout
buttons_frame = tk.Frame(window)


def hide_all_buttons():
    for widget in buttons_frame.winfo_children():
        widget.pack_forget()


def detect_image(path):
    global video_playing, progress_label
    video_playing = False

    # Clear progress/status label if exists
    if progress_label:
        progress_label.destroy()
        progress_label = None

    img = cv2.imread(path)
    results = model(img)
    rendered_img = results.render()[0]

    resized_img = cv2.resize(rendered_img, (play_width, play_height))
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb_img)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    image_label.config(image=img_tk)
    image_label.image = img_tk

    hide_all_buttons()  # Hide both play and stop
    buttons_frame.pack_forget()


def detect_video(path):
    global processed_video_path, video_playing, progress_label
    video_playing = False

    image_label.config(image="")
    image_label.image = None

    if progress_label:
        progress_label.destroy()
        progress_label = None

    hide_all_buttons()
    buttons_frame.pack_forget()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: cannot open video file {path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    processed_video_path = os.path.join("outputs", "processed_" + os.path.basename(path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    progress_label = tk.Label(window, text="Processing video: 0%")
    progress_label.pack(pady=10)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        rendered_frame = results.render()[0]
        out.write(rendered_frame)

        if i % 10 == 0 or i == total_frames - 1:
            percent = (i / total_frames) * 100
            progress_label.config(text=f"Processing video: {percent:.2f}%")
            window.update_idletasks()

    cap.release()
    out.release()

    progress_label.config(text=f"Processing complete!\nSaved to: {processed_video_path}")

    play_button.pack(side=tk.LEFT, padx=10)
    stop_button.pack(side=tk.LEFT, padx=10)
    buttons_frame.pack(pady=10)


def play_video():
    global video_capture, video_playing

    if processed_video_path is None:
        return

    if video_capture is None or not video_playing:
        video_capture = cv2.VideoCapture(processed_video_path)
        video_playing = True
        play_video_frame()


def play_video_frame():
    global video_capture, video_playing

    if not video_playing:
        if video_capture:
            video_capture.release()
        return

    ret, frame = video_capture.read()
    if not ret:
        video_playing = False
        video_capture.release()
        return

    frame = cv2.resize(frame, (play_width, play_height))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    image_label.config(image=img_tk)
    image_label.image = img_tk

    window.after(30, play_video_frame)


def stop_video():
    global video_playing
    video_playing = False


def upload_file():
    global progress_label

    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    if progress_label:
        progress_label.destroy()
        progress_label = None

    ext = file_path.lower()
    if ext.endswith((".jpg", ".jpeg", ".png")):
        detect_image(file_path)
    elif ext.endswith((".mp4", ".avi", ".mov", ".mkv")):
        detect_video(file_path)
    else:
        messagebox.showerror(
            "Unsupported File", "The selected file type is not supported. Please upload an image or video file."
        )


# Upload button (always shown)
upload_btn = tk.Button(window, text="Upload Image or Video", command=upload_file)
upload_btn.pack(pady=20)

# Play and Stop buttons (in hidden frame)
play_button = tk.Button(buttons_frame, text="Play Processed Video", command=play_video)
stop_button = tk.Button(buttons_frame, text="Stop Video", command=stop_video)

# Start the GUI
window.mainloop()
