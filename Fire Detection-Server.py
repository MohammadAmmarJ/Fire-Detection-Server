import threading
import time
import os
import cv2
import requests
import qrcode
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

def generate_qr_code(data):
    """
    Generates a QR code image from the provided data.
    :param data: The data to encode in the QR code.
    :return: A PIL.Image object representing the QR code.
    """
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        return img
    except Exception as e:
        logging.error("Failed to generate QR code", exc_info=True)
        raise e



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


def get_public_ip():
    try:
        response = requests.get('https://api.ipify.org?format=text', timeout=5)
        if response.status_code == 200:
            return response.text
    except requests.RequestException as e:
        logging.error("Failed to fetch public IP", exc_info=True)
    return "Unable to retrieve public IP"

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

    # Configure the window
    root = ctk.CTk()
    root.geometry("1600x1080")
    root.title("Fire and Smoke Detection System")
    
    # Set the color theme
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    # Create main container with padding
    main_container = ctk.CTkFrame(root, fg_color="transparent")
    main_container.pack(fill="both", expand=True, padx=20, pady=20)

    # Header Section with gradient background
    header_frame = ctk.CTkFrame(main_container, height=80, corner_radius=15)
    header_frame.pack(fill="x", pady=(0, 20))
    header_frame.pack_propagate(False)

    title_label = ctk.CTkLabel(
        header_frame, 
        text="Fire and Smoke Detection System", 
        font=("Arial Bold", 32),
        text_color="#FF6B6B"
    )
    title_label.pack(pady=20)

    # Create left and right columns
    content_frame = ctk.CTkFrame(main_container, fg_color="transparent")
    content_frame.pack(fill="both", expand=True)

    left_column = ctk.CTkFrame(content_frame, fg_color="transparent")
    left_column.pack(side="left", fill="both", expand=True, padx=(0, 10))

    right_column = ctk.CTkScrollableFrame(content_frame, fg_color="transparent", width=300)

    right_column.pack(side="right", fill="both", padx=(10, 0))

    # Network Information Section (Right Column)
    network_frame = ctk.CTkFrame(right_column, corner_radius=15)
    network_frame.pack(fill="x", pady=(0, 20))

    network_title = ctk.CTkLabel(
        network_frame,
        text="Network Information",
        font=("Arial Bold", 18),
        text_color="#4FB286"
    )
    network_title.pack(pady=(15, 10))

    ip_address = get_local_ip()
    public_ip = get_public_ip()

    network_info_frame = ctk.CTkFrame(network_frame, fg_color="transparent")
    network_info_frame.pack(pady=10, padx=15, fill="x")

    # IP Information
    for idx, (label_text, value_text) in enumerate([("Local IP:", ip_address), ("Public IP:", public_ip)]):
        ip_container = ctk.CTkFrame(network_info_frame, fg_color="transparent")
        ip_container.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            ip_container, 
            text=label_text, 
            font=("Arial", 14),
            anchor="w"
        ).pack(side="left")
        
        ctk.CTkLabel(
            ip_container, 
            text=value_text,
            font=("Arial Bold", 14),
            text_color="#6C63FF",
            anchor="e"
        ).pack(side="right")

    

    # QR Code Section
    try:
        qr_code_image = generate_qr_code(public_ip)
        qr_code_image = qr_code_image.resize((150, 150), Image.LANCZOS)
        qr_code_tk = ImageTk.PhotoImage(qr_code_image)

        qr_frame = ctk.CTkFrame(network_frame)
        qr_frame.pack(pady=(10, 15), padx=15)

        qr_code_label = ctk.CTkLabel(qr_frame, image=qr_code_tk, text="")
        qr_code_label.image = qr_code_tk
        qr_code_label.pack(pady=5)
        
        qr_text = ctk.CTkLabel(
            qr_frame, 
            text="Scan for Public IP",
            font=("Arial", 12),
            text_color="#6C63FF"
        )
        qr_text.pack()
    except Exception as e:
        logging.error("Failed to generate or display QR code", exc_info=True)
        messagebox.showerror("Error", f"Failed to generate or display QR code: {e}")

    # Video Display Section (Left Column)
    video_frame = ctk.CTkFrame(left_column, corner_radius=15)
    video_frame.pack(fill="both", expand=True)

    video_label = ctk.CTkLabel(video_frame, text=" ")
    video_label.pack(expand=True, fill='both', padx=10, pady=10)

    # Status Section
    status_frame = ctk.CTkFrame(left_column, height=80, corner_radius=15)
    status_frame.pack(fill="x", pady=(20, 0))
    status_frame.pack_propagate(False)

    detection_label = ctk.CTkLabel(
        status_frame,
        text="No Detection",
        text_color="#4FB286",
        font=("Arial Bold", 24)
    )
    detection_label.pack(expand=True)

    # System Controls Section (Right Column)
    controls_frame = ctk.CTkFrame(right_column, corner_radius=15)
    controls_frame.pack(fill="x")

    controls_title = ctk.CTkLabel(
        controls_frame,
        text="System Controls",
        font=("Arial Bold", 18),
        text_color="#4FB286"
    )
    controls_title.pack(pady=(15, 10))

    # Control Buttons
    def create_control_button(parent, text, command, state="normal", hover_color="#4FB286"):
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            state=state,
            height=40,
            corner_radius=8,
            hover_color=hover_color,
            fg_color="#2D3250"
        )

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
                video_label.configure(image='')
                video_label.image = None
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

    buttons_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
    buttons_frame.pack(pady=10, padx=15)

    start_button = create_control_button(buttons_frame, "Start System", start_system)
    start_button.pack(fill="x", pady=(0, 5))

    stop_button = create_control_button(
        buttons_frame, "Stop System", stop_system, "disabled", "#FF6B6B"
    )
    stop_button.pack(fill="x", pady=5)

    start_recording_button = create_control_button(
        buttons_frame, "Start Recording", start_recording, "disabled"
    )
    start_recording_button.pack(fill="x", pady=5)

    stop_recording_button = create_control_button(
        buttons_frame, "Stop Recording", stop_recording, "disabled", "#FF6B6B"
    )
    stop_recording_button.pack(fill="x", pady=5)

    # Source Selection Section
    source_frame = ctk.CTkFrame(right_column, corner_radius=15)
    source_frame.pack(fill="x", pady=(20, 0))

    source_title = ctk.CTkLabel(
        source_frame,
        text="Video Source Settings",
        font=("Arial Bold", 18),
        text_color="#4FB286"
    )
    source_title.pack(pady=(15, 10))

    # Source Selection Controls
    source_controls = ctk.CTkFrame(source_frame, fg_color="transparent")
    source_controls.pack(padx=15, pady=(0, 15), fill="x")

    source_var = ctk.StringVar(value='camera')
    
    radio_frame = ctk.CTkFrame(source_controls, fg_color="transparent")
    radio_frame.pack(fill="x", pady=(0, 10))

    camera_radio = ctk.CTkRadioButton(
        radio_frame,
        text="Camera",
        variable=source_var,
        value='camera',
        font=("Arial", 12)
    )
    camera_radio.pack(side="left", padx=(0, 20))

    video_radio = ctk.CTkRadioButton(
        radio_frame,
        text="Video File",
        variable=source_var,
        value='video',
        font=("Arial", 12)
    )
    video_radio.pack(side="left")

    # Camera Index Input
    camera_container = ctk.CTkFrame(source_controls, fg_color="transparent")
    camera_container.pack(fill="x", pady=5)

    ctk.CTkLabel(
        camera_container,
        text="Camera Index:",
        font=("Arial", 12)
    ).pack(side="left")

    camera_index_entry = ctk.CTkEntry(
        camera_container,
        width=50,
        height=30,
        corner_radius=8
    )
    camera_index_entry.insert(0, str(selected_camera_index))
    camera_index_entry.pack(side="right")

    # Video Path Input
    video_container = ctk.CTkFrame(source_controls, fg_color="transparent")
    video_container.pack(fill="x", pady=5)

    ctk.CTkLabel(
        video_container,
        text="Video Path:",
        font=("Arial", 12)
    ).pack(side="left")

    video_path_entry = ctk.CTkEntry(
        video_container,
        width=180,
        height=30,
        corner_radius=8
    )
    video_path_entry.insert(0, video_file_path)
    video_path_entry.pack(side="right")

    def update_video_source():
        global selected_video_source, selected_camera_index, video_file_path
        try:
            selected_video_source = source_var.get()
            selected_camera_index = int(camera_index_entry.get())
            video_file_path = video_path_entry.get()
            loading_label.configure(text="Video Source Updated", text_color="#4FB286")
        except ValueError as e:
            logging.error('Invalid video source values', exc_info=True)
            messagebox.showerror('Error', f'Invalid video source values: {e}')
            loading_label.configure(text="Failed to update video source", text_color="#FF6B6B")

    update_source_btn = create_control_button(
        source_controls,
        "Update Source",
        update_video_source
    )
    update_source_btn.pack(pady=(10, 0), fill="x")

    # Status Message Label
    loading_label = ctk.CTkLabel(
        right_column,
        text="",
        font=("Arial", 12),
        height=30
    )
    loading_label.pack(pady=(10, 0))

    def update_gui():
        global latest_frame, detection_status
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
            status = detection_status

        if running:
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Calculate the size to maintain aspect ratio
                frame_height, frame_width = frame_rgb.shape[:2]
                video_frame_width = video_frame.winfo_width() - 30  # Account for padding
                video_frame_height = video_frame.winfo_height() - 30
                
                # Calculate scaling factor to fit the frame
                width_ratio = video_frame_width / frame_width
                height_ratio = video_frame_height / frame_height
                scale_factor = min(width_ratio, height_ratio)
                
                # Calculate new dimensions
                new_width = int(frame_width * scale_factor)
                new_height = int(frame_height * scale_factor)
                
                # Resize the image while maintaining aspect ratio
                img = img.resize((new_width, new_height), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                video_label.imgtk = imgtk
                video_label.configure(image=imgtk)

            # Update detection status with color-coded text
            status_colors = {
                0: "#4FB286",  # Green for no detection
                1: "#FF6B6B",  # Red for fire
                2: "#FFB347",  # Orange for smoke
                3: "#FF6B6B"   # Red for fire and smoke
            }
            status_texts = {
                0: "No Detection",
                1: "Fire Detected!",
                2: "Smoke Detected!",
                3: "Fire and Smoke Detected!"
            }
            detection_label.configure(
                text=status_texts[status],
                text_color=status_colors[status]
            )

        root.after(100, update_gui)

    update_gui()
    root.mainloop()

if __name__ == '__main__':
    threading.Thread(target=start_video_server, daemon=True).start()
    threading.Thread(target=start_udp_server, daemon=True).start()
    create_gui()