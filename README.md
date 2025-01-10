# **Heart Rate** and **HRV Monitoring Application**
This project is a real-time heart rate and heart rate variability (HRV) monitoring application using OpenCV, MediaPipe, and Matplotlib. It employs a remote photoplethysmography (rPPG) algorithm to detect and visualize heart rate data from facial video input. The application integrates data visualization, signal processing, and interactive overlays for a modern user experience.

Features
- Real-Time Heart Rate Monitoring: Detects heart rate (BPM) from facial signals using the rPPG technique.

- Heart Rate Variability (HRV): Calculates HRV based on the interval between heartbeats.

- Interactive Visualization:
  - Displays heart rate and HRV in a sleek, real-time overlay.
  - Visualizes pulsation intensity within the region of interest (**ROI**).
- Signal Processing:
  - Applies normalization and bandpass filtering to extract meaningful signals.
  - Uses the Plane-Orthogonal-to-Skin (POS) algorithm to enhance signal clarity.
  
- Stop Functionality: Includes a **STOP** button for ***user interaction***.

**How It Works**
Face Detection:

- Uses MediaPipe FaceMesh to locate facial landmarks.
- Extracts a Region of Interest (ROI) from the forehead for signal processing.
 
Signal Processing:

- Computes average color values (RGB) from the ROI for each frame.
- Applies the POS algorithm(Plane-Orthogonal-to-Skin).<br/>

                   S(t) = 3R(t) - 2G(t)
- Filters the computed signal with a bandpass filter (**0.7–4.0 Hz**).
