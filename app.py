import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np

# YOLOv8 model load karo
model = YOLO("yolov8n.pt")  # Nano model for fast detection

# Webcam start karne ka function
def start_webcam():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam access failed!")
            break

        # YOLOv8 se detection
        results = model(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = model.names[cls]

                # Bounding box draw karo
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # BGR to RGB conversion
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()

# Streamlit UI
def main():
    st.title("🔎 YOLOv8 Real-Time Object Detection")
    st.write("Click the button below to start real-time object detection.")

    if st.button("📷 Start Webcam"):
        start_webcam()

if __name__ == "__main__":
    main()
