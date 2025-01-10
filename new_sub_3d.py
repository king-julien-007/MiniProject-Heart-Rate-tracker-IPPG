import cv2
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import mediapipe as mp
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

def calculate_heart_rate_and_hrv(ppg_signal, fps):
    filtered_signal = bandpass_filter(ppg_signal, 0.7, 4.0, fps)
    peaks, _ = find_peaks(filtered_signal, distance=fps / 2)
    
    if len(peaks) > 1:
        peak_intervals = np.diff(peaks) / fps
        heart_rate = 60 / np.mean(peak_intervals)
        hrv = np.std(peak_intervals) * 1000
    else:
        heart_rate = 0
        hrv = 0
    
    return heart_rate, hrv

def extract_ppg_from_face(frame, face_landmarks):
    h, w, _ = frame.shape
    forehead_x = int((face_landmarks[9][0] + face_landmarks[10][0]) / 2 * w)
    forehead_y = int((face_landmarks[9][1] + face_landmarks[10][1]) / 2 * h)
    roi_size = 50

    x1 = max(0, forehead_x - roi_size // 2)
    y1 = max(0, forehead_y - roi_size // 2)
    x2 = min(w, forehead_x + roi_size // 2)
    y2 = min(h, forehead_y + roi_size // 2)
    roi = frame[y1:y2, x1:x2]

    return roi, (x1, y1, x2, y2)

def create_modern_overlay(frame, heart_rate, hrv):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Create a sleek dark overlay panel
    panel_width = 300
    panel_height = 140
    panel_x = w - panel_width - 20
    panel_y = 20
    
    # Draw main panel background with gradient
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (0, 0, 0), -1)
    
    # Add accent line
    cv2.line(overlay, (panel_x, panel_y), 
             (panel_x + panel_width, panel_y), 
             (0, 255, 255), 2)
    
    # Add metrics with enhanced styling
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # BPM Display
    cv2.putText(overlay, "BPM", 
                (panel_x + 20, panel_y + 35),
                font, 0.7, (150, 150, 150), 1)
    cv2.putText(overlay, f"{heart_rate:.1f}", 
                (panel_x + 20, panel_y + 70),
                font, 1.2, (0, 255, 255), 2)
    
    # HRV Display
    cv2.putText(overlay, "HRV (ms)", 
                (panel_x + 160, panel_y + 35),
                font, 0.7, (150, 150, 150), 1)
    cv2.putText(overlay, f"{hrv:.1f}", 
                (panel_x + 160, panel_y + 70),
                font, 1.2, (0, 255, 255), 2)
    
    # Add status indicator
    status_color = (0, 255, 0) if heart_rate > 0 else (150, 150, 150)
    cv2.circle(overlay, (panel_x + panel_width - 20, panel_y + 20), 
               5, status_color, -1)
    
    # Blend the overlay with the original frame
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame

def init_plot():
    plt.ion()
    fig = plt.figure(figsize=(12, 4), facecolor='black')
    ax = fig.add_subplot(111, facecolor='black')
    
    # Create empty line objects with enhanced styling
    line, = ax.plot([], [], color='#00ff00', linewidth=1.5, label='Heart Rate')
    peaks_scatter = ax.scatter([], [], color='#ff3366', marker='x', s=100, label='Peaks')
    
    # Customize the plot with a more professional look
    ax.set_title('Real-time Heart Rate Monitor', color='white', pad=10, fontsize=12)
    ax.set_xlabel('Time (s)', color='#888888', fontsize=10)
    ax.set_ylabel('BPM', color='#888888', fontsize=10)
    
    # Set fixed y-axis limits
    ax.set_ylim(40, 120)
    
    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    
    # Customize spine colors
    for spine in ax.spines.values():
        spine.set_color('#444444')
    
    # Customize tick colors
    ax.tick_params(colors='#888888')
    
    # Add horizontal reference lines
    ax.axhline(y=60, color='#444444', linestyle='--', alpha=0.5)
    ax.axhline(y=100, color='#444444', linestyle='--', alpha=0.5)
    
    # Customize legend
    ax.legend(loc='upper right', facecolor='black', edgecolor='#444444', 
             labelcolor='white', fontsize=8)
    
    plt.tight_layout()
    return fig, ax, line, peaks_scatter

def update_plot(line, peaks_scatter, ax, times, values):
    if len(times) == 0:
        return
    
    # Convert to numpy array for easier manipulation
    times = np.array(times)
    values = np.array(values)
    
    # Find peaks in the current window
    peaks, _ = find_peaks(values, distance=10, prominence=1)
    
    # Update the main line
    line.set_data(times, values)
    
    # Update peaks
    if len(peaks) > 0:
        peaks_scatter.set_offsets(np.column_stack((times[peaks], values[peaks])))
    else:
        peaks_scatter.set_offsets(np.empty((0, 2)))
    
    # Update x-axis to show continuous scrolling
    current_time = times[-1]
    ax.set_xlim(current_time - 30, current_time)
    
    plt.draw()
    plt.pause(0.01)

# Flag to stop the program
stop_program = False

def stop_button_callback(event, x, y, flags, param):
    global stop_program
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 110 and 10 <= y <= 60:  # Coordinates of the stop button
            stop_program = True

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Error: Unable to open webcam.")

    # Initialize face mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    # Initialize variables
    fps = 30
    raw_R, raw_G, raw_B = [], [], []
    
    # Initialize plotting with larger deques for continuous scrolling
    fig, ax, line, peaks_scatter = init_plot()
    bpm_values = deque(maxlen=None)  # No maximum length for continuous scrolling
    time_points = deque(maxlen=None)
    current_time = 0

    cv2.namedWindow("Heart Rate Monitor")
    cv2.setMouseCallback("Heart Rate Monitor", stop_button_callback)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or stop_program:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_frame)

            heart_rate, hrv = 0, 0

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    # Draw face bounding box with enhanced style
                    face_points = np.array([(lm.x * frame.shape[1], lm.y * frame.shape[0]) 
                                          for lm in face_landmarks.landmark])
                    face_bbox = np.int32([
                        [np.min(face_points[:, 0]), np.min(face_points[:, 1])],
                        [np.max(face_points[:, 0]), np.max(face_points[:, 1])]
                    ])
                    
                    # Add glowing effect to face detection box
                    cv2.rectangle(frame, 
                                (face_bbox[0][0]-2, face_bbox[0][1]-2),
                                (face_bbox[1][0]+2, face_bbox[1][1]+2),
                                (0, 255, 255), 3)
                    cv2.rectangle(frame, 
                                (face_bbox[0][0], face_bbox[0][1]),
                                (face_bbox[1][0], face_bbox[1][1]),
                                (0, 0, 0), 1)

                    # Extract ROI and draw ROI box with enhanced style
                    landmark_coords = [(lm.x, lm.y) for lm in face_landmarks.landmark]
                    roi, roi_box = extract_ppg_from_face(frame, landmark_coords)
                    
                    # Add glowing effect to ROI box
                    cv2.rectangle(frame, 
                                (roi_box[0]-1, roi_box[1]-1),
                                (roi_box[2]+1, roi_box[3]+1),
                                (255, 255, 255), 2)
                    cv2.rectangle(frame, 
                                (roi_box[0], roi_box[1]),
                                (roi_box[2], roi_box[3]),
                                (0, 0, 0), 1)

                    # Process colors and calculate heart rate
                    avg_color = np.mean(roi, axis=(0, 1))
                    raw_B.append(avg_color[0])
                    raw_G.append(avg_color[1])
                    raw_R.append(avg_color[2])

                    if len(raw_R) > fps * 10:
                        raw_R.pop(0)
                        raw_G.pop(0)
                        raw_B.pop(0)

                    if len(raw_R) >= fps * 5:
                        R_norm = np.array(raw_R)
                        G_norm = np.array(raw_G)
                        B_norm = np.array(raw_B)
                        
                        R_norm = (R_norm - np.mean(R_norm)) / np.std(R_norm)
                        G_norm = (G_norm - np.mean(G_norm)) / np.std(G_norm)
                        B_norm = (B_norm - np.mean(B_norm)) / np.std(B_norm)

                        S = 3 * R_norm - 2 * G_norm
                        heart_rate, hrv = calculate_heart_rate_and_hrv(S, fps)

                        if heart_rate > 0:
                            bpm_values.append(heart_rate)
                            time_points.append(current_time)
                            update_plot(line, peaks_scatter, ax, 
                                      list(time_points), 
                                      list(bpm_values))
                            current_time += 1/fps

            # Apply enhanced overlay
            frame = create_modern_overlay(frame, heart_rate, hrv)

            # Draw the stop button
            cv2.rectangle(frame, (10, 10), (110, 60), (0, 0, 255), -1)
            cv2.putText(frame, "STOP", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Heart Rate Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.close()