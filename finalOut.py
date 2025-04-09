import cv2
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import mediapipe as mp
from collections import deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import geocoder
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.animation import FuncAnimation

HEART_RATE_THRESHOLD = 60
EMAIL_COOLDOWN_SECONDS = 300
LOW_HEART_RATE_DURATION = 15
SENDER_EMAIL = "gayathriss2025@gmail.com"
RECEIVER_EMAILS = ["abhilaabhi18@gmail.com", "r.ardrasankar@gmail.com"]
EMAIL_PASSWORD = "owioyearcgjaxpej"

last_alert_time = 0
low_heart_rate_start_time = None
emergency_message = ""
emergency_message_timer = 0

heart_rate_history = deque(maxlen=300)  
time_history = deque(maxlen=300)       
start_time = time.time()              

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

def calculate_heart_rate_and_hrv(ppg_signal, fps):
    try:
        filtered_signal = bandpass_filter(ppg_signal, 0.7, 4.0, fps)
        peaks, _ = find_peaks(filtered_signal, distance=fps // 2)
        
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / fps
            heart_rate = 60 / np.mean(peak_intervals)
            hrv = np.std(peak_intervals) * 1000
        else:
            heart_rate = 0
            hrv = 0
        
        return heart_rate, hrv
    except Exception as e:
        print(f"Heart rate calculation error: {e}")
        return 0, 0

def extract_ppg_from_face(frame, landmarks):
    h, w, _ = frame.shape
    try:
        x = int((landmarks[9][0] + landmarks[10][0]) / 2 * w)
        y = int((landmarks[9][1] + landmarks[10][1]) / 2 * h)
        size = 50
        x1, y1 = max(0, x - size // 2), max(0, y - size // 2)
        x2, y2 = min(w, x + size // 2), min(h, y + size // 2)
        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)
    except Exception as e:
        print(f"ROI extraction error: {e}")
        return None, None

def send_fatigue_alert():
    global last_alert_time, emergency_message, emergency_message_timer
    
    try:
        now = time.time()
        loc = geocoder.ip("me").latlng
        location_text = f"Location: https://www.google.com/maps?q={loc[0]},{loc[1]}" if loc else "Location unavailable."

        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = ", ".join(RECEIVER_EMAILS)
        msg["Subject"] = "DRIVER FATIGUE EMERGENCY ALERT"
        body = f"EMERGENCY: Low Heart Rate Detected\n\n{location_text}"
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())
        
        emergency_message = "EMERGENCY ALERT SENT!"
        emergency_message_timer = 300
        print("Emergency alert sent successfully!")
    except Exception as e:
        print(f"Alert sending failed: {e}")

def create_overlay(frame, heart_rate, hrv, threshold):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # UI design ivda muthal
    panel_width, panel_height = 300, 120
    cv2.rectangle(overlay, (w-panel_width-20, 20), 
                  (w-20, 20+panel_height), 
                  (0, 0, 0), -1)
    
    # BPM ivda 
    cv2.putText(overlay, f"BPM: {heart_rate:.1f}", 
                (w-280, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (0, 255, 255), 2)
    
    # Threshold marker red clr
    status_color = (0, 255, 0) if heart_rate >= threshold else (0, 0, 255)
    cv2.putText(overlay, f"Threshold: {threshold}", 
                (w-280, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                status_color, 1)
    
    # Emergency text valeu
    global emergency_message, emergency_message_timer
    if emergency_message_timer > 0:
        cv2.putText(overlay, emergency_message, 
                    (50, h-50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 0, 255), 2)
        emergency_message_timer -= 1
        if emergency_message_timer == 0:
            emergency_message = ""
    
    return overlay

def update_graph(frame, ax, line):
    """Update the graph with new data."""
    ax.clear()
    ax.set_title("Heart Rate Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart Rate (BPM)")
    ax.set_ylim(0, 150)  # Adjustable option
    ax.set_xlim(max(0, time.time() - start_time - 300), time.time() - start_time)  # Show last 300 seconds
    ax.grid(True)

    # Plot the heart shit
    line.set_data(time_history, heart_rate_history)
    ax.plot(time_history, heart_rate_history, color="blue")
    return line,

def setup_graph():
    """Initialize the graph."""
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ani = FuncAnimation(fig, update_graph, fargs=(ax, line), interval=1000)
    plt.show(block=False)
    return ani

def main():
    global low_heart_rate_start_time, HEART_RATE_THRESHOLD, EMAIL_COOLDOWN_SECONDS, heart_rate_history, time_history

    
    cv2.namedWindow("Settings")
    cv2.createTrackbar("Threshold", "Settings", 60, 100, lambda x: globals().update(HEART_RATE_THRESHOLD=x))
    cv2.createTrackbar("Cooldown (sec)", "Settings", 30, 600, lambda x: globals().update(EMAIL_COOLDOWN_SECONDS=x*10))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  
    
    raw_colors = {'R': [], 'G': [], 'B': []}
    last_alert_time = 0

    ani = setup_graph()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        results = face_mesh.process(rgb_frame)
        
        heart_rate, hrv = 0, 0
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmark_coords = [(lm.x, lm.y) for lm in face_landmarks.landmark]
                roi, _ = extract_ppg_from_face(frame, landmark_coords)
                
                if roi is not None:
                    avg_color = np.mean(roi, axis=(0, 1))
                    
                    for color, val in zip(['B', 'G', 'R'], avg_color):
                        raw_colors[color].append(val)
                        if len(raw_colors[color]) > fps * 10:
                            raw_colors[color].pop(0)

                    if len(raw_colors['R']) >= fps * 5:
                        signals = {color: (np.array(raw_colors[color]) - np.mean(raw_colors[color])) / np.std(raw_colors[color]) 
                                   for color in ['R', 'G', 'B']}
                        
                        combined_signal = 3 * signals['R'] - 2 * signals['G']
                        
                        heart_rate, hrv = calculate_heart_rate_and_hrv(combined_signal, fps)

                        if heart_rate < HEART_RATE_THRESHOLD:
                            if low_heart_rate_start_time is None:
                                low_heart_rate_start_time = time.time()
                            elif time.time() - low_heart_rate_start_time > LOW_HEART_RATE_DURATION:
                                if time.time() - last_alert_time > EMAIL_COOLDOWN_SECONDS:
                                    send_fatigue_alert()
                                    last_alert_time = time.time()
                                    low_heart_rate_start_time = None
                        else:
                            low_heart_rate_start_time = None

        current_time = time.time() - start_time
        heart_rate_history.append(heart_rate)
        time_history.append(current_time)

        overlay_frame = create_overlay(frame, heart_rate, hrv, HEART_RATE_THRESHOLD)
        
        cv2.imshow("Fatigue Monitor", overlay_frame)
        
        #exit press q or click the crossbar 
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # exit
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
