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
1. Face Detection:

- Uses MediaPipe FaceMesh to locate facial landmarks.
- Extracts a Region of Interest (ROI) from the forehead for signal processing.
 
2. Signal Processing:

- Computes average color values (RGB) from the ROI for each frame.
- Applies the POS algorithm(Plane-Orthogonal-to-Skin).<br/>

                   S(t) = 3R(t) - 2G(t)
- Filters the computed signal with a bandpass filter (**0.7–4.0 Hz**).

3. Heart Rate and HRV Calculation:

- Detects peaks in the filtered signal to calculate heart rate (**BPM**).
- Computes HRV as the standard deviation of the intervals between peaks.

4. Visualization:

- Displays heart rate and HRV metrics in a **real-time overlay**.
- Dynamically visualizes pulsation intensity on the ***ROI***.

5. Interactive Plot:

- Plots heart rate values over time using **Matplotlib**.

**Installation**
1. Clone the repository:<br/>

                          git clone https://github.com/your-username/heart-rate-monitor.git
   
                          cd heart-rate-monitor
3. Install dependencies:
     ```python
                          pip install -r requirements.txt
  - The dependencies include:
    - `opencv-python`
    - `mediapipe`
    - `matplotlib`
    - `numpy`
    - `scipy`
--------------------------------------------------------------------------------------------------------------

4. Implementation:

  - Ensure your webcam is enabled.
  - The application will display a live video feed with overlaid heart rate and HRV values.
  - Press the STOP button or q to exit.


**Acknowledgments**

    MediaPipe: For face detection and landmark tracking.
    OpenCV: For video processing.
    Matplotlib: For real-time visualization.




