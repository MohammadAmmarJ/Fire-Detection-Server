# Fire-Detection-Server

## 🚀 Overview
The **Fire-Detection-Server** is the core backend system for the **Fire & Smoke Detection System**, utilizing AI-powered YOLO models to detect fire and smoke in real-time. It processes video streams, provides detection alerts, and enables seamless communication with client applications.

## 📌 Features
- 🔥 **Real-Time AI Detection** – Uses YOLO-based deep learning models.
- 🎥 **Live Video Streaming** – Streams annotated video with detected fire and smoke.
- 🌍 **Network Connectivity** – Supports both LAN and WAN with auto-discovery.
- 🔔 **Instant Notifications** – Sends alerts for fire and smoke detection.
- 📡 **Lightweight & Efficient** – Optimized for real-time performance.

## 🏗️ System Architecture
1️⃣ **Camera Captures Video** 🎥  
2️⃣ **Server with YOLO detects fire/smoke** 🔥  
3️⃣ **Annotated frames & status codes generated** 💾  
4️⃣ **Client app receives video & alerts user** 📲  

## 🛠️ Technologies Used
- **Programming Language:** Python (Flask / FastAPI)
- **Machine Learning Model:** YOLO-based fire and smoke detection (YOLO11m)
- **Network Protocols:** UDP Broadcast for LAN discovery, HTTP for API communication
- **Client App:** .NET MAUI for real-time mobile monitoring

## 📦 Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/MohammadAmmarJ/Fire-Detection-Server
   cd Fire-Detection-Server
   ```
2. Set up a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## 🚀 Running the Server
Start the app with:
```sh
py "Fire Detection-Server.py"
```

## 🌍 Network Discovery
- **LAN Mode:** Uses UDP broadcast on port `9999` to auto-discover the server.
- **WAN Mode:** Clients can scan a QR code to connect to the remote server.


## 📱 Client Application (Fire-Detection-App)
The **Fire-Detection-App** (built with .NET MAUI) enables users to:
- View live annotated video feeds.
- Receive instant notifications.
- Switch between LAN and WAN modes.
  
Available on : [Fire-Detection Client Application](https://github.com/MohammadAmmarJ/Fire-Detection-Client)

## 📞 Contact

- 📧 Email: [mohammadammarga@gmail.com](mailto\:mohammadammarga@gmail.com)
- 📧 Email: [amroadnanb@gmail.com](mailto\:amroadnanb@gmail.com)
---
