# Virtual Steering Wheel 🏎️👐

Turn your hands into a virtual steering wheel using your webcam! This project uses computer vision to track your hands and translates the angle of your hands into keyboard steering inputs (`A`/`D` or Arrow keys) to control racing games or other applications.

Built using **Python**, **OpenCV**, and the modern **MediaPipe Tasks API**.

---

## How It Works 🛠️

1. **Webcam Input**: Captures real-time frames from your webcam.
2. **Hand Tracking**: Uses the modern MediaPipe Hand Landmarker model to detect and track both of your hands.
3. **Angle Calculation**: Calculates the angle between the centroids of your left and right hands.
4. **Smoothing & Dead Zone**: Filters the steering angle to prevent hand jitter, applying a dead zone to make driving straight easier.
5. **Keyboard Emulation**: Uses the `keyboard` library to press and hold keys based on the calculated steering angle.

---

## Controls 🕹️

* **`c`** - **Calibrate**: Set your current hand position as "straight up" / center.
* **`m`** - **Toggle Output Mode**: Switch between `A`/`D` keys and `Left`/`Right` arrow keys.
* **`q`** - **Quit**: Safely close the camera stream and exit the program.

---

## Installation 📦

1. Clone this repository:
   ```bash
   git clone https://github.com/joyccceee-keren/Virtual_-sterring.git
   cd Virtual_-sterring
   ```

2. Install the required dependencies:
   ```bash
   pip install opencv-python mediapipe keyboard
   ```

---

## Running the Application 🚀

Run the script using:
```bash
python virtual_steering_wheel.py
```

> [!IMPORTANT]
> **Windows Admin Rights**: Because the script simulates global hardware keypresses via the `keyboard` library, you should run your terminal (Command Prompt or PowerShell) **as Administrator**. Otherwise, games may not register the keyboard inputs.

> [!NOTE]
> On the first run, the script will automatically download the necessary MediaPipe model file (`hand_landmarker.task` ~5.6MB) directly from Google's official repositories.
