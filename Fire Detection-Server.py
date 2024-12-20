import threading
import time
import os
import cv2
import requests
import customtkinter as ctk
from flask import Flask, Response, request, jsonify
from ultralytics import YOLO
from datetime import datetime
from PIL import Image, ImageTk
from queue import Queue, Empty
from flask_cors import CORS
import socket
import logging
import tkinter.messagebox as messagebox  # For displaying error dialogs

# Configure logging to log application events and errors
logging.basicConfig(
    filename='application.log',  # Log file name
    level=logging.DEBUG,         # Log level
    format='%(asctime)s %(levelname)s:%(message)s'  # Log format
)

# Initialize a Flask web application
video_app = Flask(__name__)
CORS(video_app)  # Enable Cross-Origin Resource Sharing (CORS)

# Initialize YOLO object detection model
try:
    model = YOLO("Fire-Best.pt")  # Load a pre-trained YOLO model for fire and smoke detection
except Exception as e:
    logging.error("Failed to load YOLO model", exc_info=True)  # Log errors if model fails to load
    raise e

# Global variables for managing application state
running = False  # Indicates whether the system is running

detection_status = 0  # Detection states: 0: No detection, 1: Fire, 2: Smoke, 3: Fire and Smoke
last_notification_time = 0  # Timestamp of the last notification sent
frame_queue = Queue(maxsize=5)  # Queue for storing frames for processing
video_writer = None  # Video writer for saving recorded videos
record_video = False  # Flag indicating whether video recording is active
lock = threading.Lock()  # Lock for synchronizing access to shared resources
latest_frame = None  # The latest processed video frame
selected_camera_index = 0  # Default camera index for video capture
selected_video_source = 'camera'  # Indicates whether source is a camera or video file
video_file_path = ''  

# **Detection confidence fixed at 60%**
detection_confidence = 0.6

# UDP Discovery settings
DISCOVERY_PORT = 9999  # UDP port for discovery messages
DISCOVERY_MESSAGE = b'FIRE_DETECTION_SERVER_DISCOVERY'  # Message for client discovery
RESPONSE_MESSAGE = b'FIRE_DETECTION_SERVER_RESPONSE'    # Server response to discovery

# Ensure directories for saving images and videos exist
os.makedirs("detections", exist_ok=True)  # Directory for saving detection images
os.makedirs("videos", exist_ok=True)      # Directory for saving recorded videos

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to an arbitrary unreachable IP to determine local IP
        s.connect(('10.255.255.255', 1))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = '127.0.0.1'  # Default to localhost if unable to determine IP
    finally:
        s.close()
    return local_ip

def save_detection_image(frame):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detections/detection_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        logging.info(f"Saved detection image: {filename}")
    except Exception as e:
        logging.error("Failed to save detection image", exc_info=True)

def get_detection_message():
    if detection_status == 1:
        return "Fire Detected!"
    elif detection_status == 2:
        return "Smoke Detected!"
    elif detection_status == 3:
        return "Fire and Smoke Detected!"
    else:
        return "No Detection"

def start_udp_server():
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp_sock.bind(('', DISCOVERY_PORT))

    while True:
        try:
            data, addr = udp_sock.recvfrom(1024)
            if data == DISCOVERY_MESSAGE:
                server_ip = get_local_ip()
                response = RESPONSE_MESSAGE + server_ip.encode('utf-8')
                udp_sock.sendto(response, addr)
                logging.info(f"Sent discovery response to {addr}")
        except Exception as e:
            logging.error("Error in UDP discovery server", exc_info=True)

class VideoProcessor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = False

    def run(self):
        global video_writer, record_video
        try:
            self.running = True
            self.initialize_capture()

            while self.running:
                success, frame = self.cap.read()
                if not success:
                    logging.warning("Failed to read frame from video source")
                    time.sleep(0.1)
                    continue

                frame_resized = cv2.resize(frame, (640, 480))

                if not frame_queue.full():
                    frame_queue.put(frame_resized)

                if record_video and video_writer is not None:
                    video_writer.write(frame)

                time.sleep(0.03)  # Approximate 30 FPS

        except Exception as e:
            logging.error("Error in VideoProcessor", exc_info=True)
        finally:
            if self.cap is not None:
                self.cap.release()
            if video_writer is not None:
                video_writer.release()

    def initialize_capture(self):
        try:
            if selected_video_source == 'camera':
                self.cap = cv2.VideoCapture(selected_camera_index)
            else:
                self.cap = cv2.VideoCapture(video_file_path)
            logging.info(f"Video source initialized: {selected_video_source}")
        except Exception as e:
            logging.error("Failed to initialize video capture", exc_info=True)
            raise e

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()

class FrameProcessor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = False

    def run(self):
        global latest_frame, detection_status
        try:
            self.running = True

            while self.running:
                try:
                    frame = frame_queue.get(timeout=1)
                except Empty:
                    continue

                results = model(frame, show=False, classes=[0, 2], conf=detection_confidence)
                annotated_frame = results[0].plot()

                fire_detected = any(box.cls == 0 for box in results[0].boxes)
                smoke_detected = any(box.cls == 2 for box in results[0].boxes)

                with lock:
                    if fire_detected and smoke_detected:
                        detection_status = 3
                    elif fire_detected:
                        detection_status = 1
                    elif smoke_detected:
                        detection_status = 2
                    else:
                        detection_status = 0

                    latest_frame = annotated_frame.copy()

                if detection_status != 0:
                    threading.Thread(target=save_detection_image, args=(annotated_frame.copy(),)).start()

        except Exception as e:
            logging.error("Error in FrameProcessor", exc_info=True)

    def stop(self):
        self.running = False

def video_feed():
    def generate():
        global latest_frame
        try:
            while running:
                with lock:
                    frame = latest_frame.copy() if latest_frame is not None else None
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    time.sleep(0.1)
        except GeneratorExit:
            logging.info("Video feed generator closed")
        except Exception as e:
            logging.error("Error in video_feed generator", exc_info=True)

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def notification_status_route():
    return str(detection_status)

def start_video_server():
    try:
        video_app.add_url_rule('/video_feed', 'video_feed', video_feed)
        video_app.add_url_rule('/status', 'notification_status_route', notification_status_route)
        video_app.run(host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        logging.error("Failed to start Flask server", exc_info=True)

def create_gui():
    global running, video_writer, record_video, latest_frame, selected_camera_index
    global selected_video_source, video_file_path

    video_thread = None
    frame_processor_thread = None

    root = ctk.CTk()
    root.geometry("800x800")
    root.title("Fire and Smoke Detection System")

    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    ip_address = get_local_ip()

    ip_label = ctk.CTkLabel(root, text=f"Server IP Address: {ip_address}", font=("Arial", 16))
    ip_label.pack(pady=5)

    video_label = ctk.CTkLabel(root, text=" ")
    video_label.pack(expand=True, fill='both')

    detection_label = ctk.CTkLabel(root, text="No Detection", text_color="green", font=("Arial", 24))
    detection_label.pack(pady=10)

    loading_label = ctk.CTkLabel(root, text="")
    loading_label.pack(pady=5)

    control_frame = ctk.CTkFrame(root)
    control_frame.pack(pady=10)

    def start_system():
        global running
        nonlocal video_thread, frame_processor_thread
        if not running:
            try:
                running = True
                video_thread = VideoProcessor()
                frame_processor_thread = FrameProcessor()
                video_thread.start()
                frame_processor_thread.start()
                start_button.configure(state="disabled")
                stop_button.configure(state="normal")
                start_recording_button.configure(state="normal")
                stop_recording_button.configure(state="normal")
                loading_label.configure(text="System Running...", text_color="green")
            except Exception as e:
                logging.error('Failed to start the system', exc_info=True)
                messagebox.showerror('Error', f'Failed to start the system: {e}')
                loading_label.configure(text="Failed to start the system", text_color="red")

    def stop_system():
        global running
        nonlocal video_thread, frame_processor_thread
        if running:
            try:
                running = False
                video_thread.stop()
                frame_processor_thread.stop()
                video_thread.join()
                frame_processor_thread.join()
                start_button.configure(state="normal")
                stop_button.configure(state="disabled")
                start_recording_button.configure(state="disabled")
                stop_recording_button.configure(state="disabled")
                loading_label.configure(text="System Stopped", text_color="red")
            except Exception as e:
                logging.error('Failed to stop the system', exc_info=True)
                messagebox.showerror('Error', f'Failed to stop the system: {e}')

    def start_recording():
        global video_writer, record_video
        if video_writer is None and video_thread is not None:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"videos/recording_{timestamp}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                frame_width = int(video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))
                record_video = True
                loading_label.configure(text="Recording Started", text_color="green")
            except Exception as e:
                logging.error('Failed to start recording', exc_info=True)
                messagebox.showerror('Error', f'Failed to start recording: {e}')

    def stop_recording():
        global video_writer, record_video
        if video_writer is not None:
            try:
                record_video = False
                video_writer.release()
                video_writer = None
                loading_label.configure(text="Recording Stopped", text_color="red")
            except Exception as e:
                logging.error('Failed to stop recording', exc_info=True)
                messagebox.showerror('Error', f'Failed to stop recording: {e}')

    start_button = ctk.CTkButton(control_frame, text="Start System", command=start_system)
    start_button.grid(row=0, column=0, padx=10)

    stop_button = ctk.CTkButton(control_frame, text="Stop System", command=stop_system, state="disabled")
    stop_button.grid(row=0, column=1, padx=10)

    start_recording_button = ctk.CTkButton(control_frame, text="Start Recording", command=start_recording, state="disabled")
    start_recording_button.grid(row=0, column=2, padx=10)

    stop_recording_button = ctk.CTkButton(control_frame, text="Stop Recording", command=stop_recording, state="disabled")
    stop_recording_button.grid(row=0, column=3, padx=10)

    source_frame = ctk.CTkFrame(root)
    source_frame.pack(pady=10)

    source_label = ctk.CTkLabel(source_frame, text="Video Source:")
    source_label.grid(row=0, column=0, padx=5)

    source_var = ctk.StringVar(value='camera')
    camera_radio = ctk.CTkRadioButton(source_frame, text="Camera", variable=source_var, value='camera')
    camera_radio.grid(row=0, column=1, padx=5)
    video_radio = ctk.CTkRadioButton(source_frame, text="Video File", variable=source_var, value='video')
    video_radio.grid(row=0, column=2, padx=5)

    camera_index_label = ctk.CTkLabel(source_frame, text="Camera Index:")
    camera_index_label.grid(row=1, column=0, padx=5)
    camera_index_entry = ctk.CTkEntry(source_frame, width=50)
    camera_index_entry.insert(0, str(selected_camera_index))
    camera_index_entry.grid(row=1, column=1, padx=5)

    video_path_label = ctk.CTkLabel(source_frame, text="Video File Path:")
    video_path_label.grid(row=1, column=2, padx=5)
    video_path_entry = ctk.CTkEntry(source_frame, width=200)
    video_path_entry.insert(0, video_file_path)
    video_path_entry.grid(row=1, column=3, padx=5)

    def update_video_source():
        global selected_video_source, selected_camera_index, video_file_path
        try:
            selected_video_source = source_var.get()
            selected_camera_index = int(camera_index_entry.get())
            video_file_path = video_path_entry.get()
            loading_label.configure(text="Video Source Updated", text_color="green")
        except ValueError as e:
            logging.error('Invalid video source values', exc_info=True)
            messagebox.showerror('Error', f'Invalid video source values: {e}')
            loading_label.configure(text="Failed to update video source", text_color="red")

    source_update_button = ctk.CTkButton(source_frame, text="Update Source", command=update_video_source)
    source_update_button.grid(row=2, column=0, columnspan=4, pady=5)

    def update_gui():
        global latest_frame, detection_status
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
            status = detection_status

        if running:
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                video_label.imgtk = imgtk
                video_label.configure(image=imgtk)

            # Update GUI detection label
            if status == 1:
                detection_label.configure(text="Fire Detected!", text_color="red")
            elif status == 2:
                detection_label.configure(text="Smoke Detected!", text_color="orange")
            elif status == 3:
                detection_label.configure(text="Fire and Smoke Detected!", text_color="red")
            else:
                detection_label.configure(text="No Detection", text_color="green")

        root.after(100, update_gui)

    update_gui()
    root.mainloop()

if __name__ == '__main__':
    threading.Thread(target=start_video_server, daemon=True).start()
    threading.Thread(target=start_udp_server, daemon=True).start()
    create_gui()
