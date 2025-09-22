# Driver Distraction Monitoring System

A Python project for detecting driver distraction using both computer vision (webcam) and gyroscope sensor data. Includes a real-time dashboard, model training scripts, and sample data.

## Features
- **Streamlit Dashboard:** Simulate steering and sensor data, visualize distraction prediction in real time.
- **Webcam Monitoring:** Detects face and eyes using OpenCV; alerts if the driver is distracted.
- **Machine Learning Model:** Trains a Random Forest classifier on gyroscope data to classify driver focus.
- **Sample Data:** Includes raw and labeled gyroscope data for experimentation.

## File Overview
- `dashboard.py` — Streamlit app for real-time simulation and prediction.
- `driver_monitor.py` — Webcam-based distraction detection.
- `Test.py` — Data processing, feature extraction, model training, and evaluation.
- `driver_distraction_model.pkl` — Trained ML model for distraction prediction.
- `phone_data.csv` — Raw gyroscope data.
- `labeled_data.csv` — Labeled gyroscope data.
- `requirements.txt` — Python dependencies.
- `packages.txt` — System dependencies (e.g., for OpenCV).

## Getting Started
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   For OpenCV on some systems, you may also need:
   ```bash
   sudo apt-get install libgl1
   ```
2. **Run the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```
3. **Run webcam monitoring:**
   ```bash
   python driver_monitor.py
   ```
4. **Train the model (optional):**
   ```bash
   python Test.py
   ```

## License
MIT License
