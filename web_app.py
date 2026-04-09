import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="SmartGuard AI", layout="centered")

st.title("🌐 SMART OBJECT DETECTION SYSTEM")
st.write("Click the button below to activate the AI Guard")

# --- START BUTTON ---
if st.button('ACTIVATE CAMERA'):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)
    
    # Empty placeholder for camera feed
    st_frame = st.empty()
    stop_btn = st.button("STOP DETECTION")

    while cap.isOpened():
        success, frame = cap.read()
        if not success or stop_btn:
            break

        results = model(frame, conf=0.30, stream=True)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cls = int(box.cls[0])
                name = model.names[cls].lower()
                
                # UNGA LOGIC MAPPING
                display_name = name.upper()
                if name in ['kite', 'airplane']: display_name = "FAN"
                elif name in ['toothbrush', 'knife']: display_name = "PEN"
                elif name == 'cell phone':
                    display_name = "ID CARD" if h > w else "MOBILE PHONE"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, display_name, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Streamlit-la frame-a display panna:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()