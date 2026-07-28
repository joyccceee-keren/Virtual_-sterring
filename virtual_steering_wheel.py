"""
Virtual Steering Wheel with Arcade Game
--------------------------------------
Turns your two hands (tracked via webcam) into a steering wheel.
Left side (640x480): Camera feed with hand skeleton tracking.
Right side (640x480): A scrolling 2D arcade car game controlled by steering.

Controls:
  c   - calibrate (set current hand angle as center/straight)
  m   - toggle output mode: keys (a/d) <-> arrows (left/right)
  r   - restart the 2D game
  q   - quit

Install dependencies first:
  pip install opencv-python mediapipe keyboard numpy --break-system-packages
"""

import cv2
import mediapipe as mp
import math
import time
import sys
import os
import urllib.request
import random
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except Exception as e:
    KEYBOARD_AVAILABLE = False
    print("[WARN] 'keyboard' module not available or lacks permission.")
    print(f"       Reason: {e}")
    print("       Script will still run and show the virtual wheel, but won't send key presses.")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEAD_ZONE_DEG = 6          # +/- degrees around center that count as "straight"
FULL_LEFT_DEG = 35         # angle (from center) at which we call it "full left"
FULL_RIGHT_DEG = 35        # angle (from center) at which we call it "full right"
SMOOTHING = 0.25           # 0-1, higher = more responsive, lower = smoother/slower
MIN_HAND_DETECTION_CONF = 0.6
MIN_HAND_TRACKING_CONF = 0.6

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# ---------------------------------------------------------------------------
# Custom Hand Drawing Constants & Connections
# ---------------------------------------------------------------------------
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17), (0, 17)
]

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
center_offset_deg = 0.0      # calibration offset
smoothed_angle = 0.0         # filtered wheel angle relative to center
output_mode = "keys"         # "keys" (a/d) or "arrows" (left/right)
current_key_held = None      # tracks which key is currently pressed down

# ---------------------------------------------------------------------------
# Game State
# ---------------------------------------------------------------------------
game_active = True
score = 0
high_score = 0
speed = 5
player_x = 320
player_y = 390
obstacles = []
last_spawn_time = 0.0
road_scroll = 0


def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Downloading hand landmarker model from {MODEL_URL}...")
        try:
            def progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                sys.stdout.write(f"\rDownloading model... {percent:.1f}%")
                sys.stdout.flush()

            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, progress)
            print("\n[INFO] Model downloaded successfully.")
        except Exception as e:
            print(f"\n[ERROR] Failed to download model: {e}")
            sys.exit(1)


def draw_hand_landmarks(frame, hand_landmarks, w, h):
    """Draw a styled hand skeleton using OpenCV."""
    points = []
    for lm in hand_landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))

    # Draw connection lines
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], (200, 200, 200), 2)

    # Draw joints/landmarks
    for idx, (px, py) in enumerate(points):
        if idx in (4, 8, 12, 16, 20):  # Finger Tips
            cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)  # Neon Green
        elif idx == 0:  # Wrist
            cv2.circle(frame, (px, py), 8, (0, 165, 255), -1)  # Orange
        else:  # Other joints
            cv2.circle(frame, (px, py), 4, (255, 200, 0), -1)  # Cyan


def hand_center(landmarks, frame_w, frame_h):
    """Average of all landmark points for one hand -> (x, y) in pixels."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    cx = sum(xs) / len(xs) * frame_w
    cy = sum(ys) / len(ys) * frame_h
    return cx, cy


def release_all_keys():
    global current_key_held
    if not KEYBOARD_AVAILABLE:
        return
    for k in ("a", "d", "left", "right"):
        keyboard.release(k)
    current_key_held = None


def send_steering_output(angle_deg):
    """Given the calibrated, smoothed angle, press the correct key."""
    global current_key_held

    if not KEYBOARD_AVAILABLE:
        return

    left_key = "a" if output_mode == "keys" else "left"
    right_key = "d" if output_mode == "keys" else "right"

    if angle_deg < -DEAD_ZONE_DEG:
        desired = left_key
    elif angle_deg > DEAD_ZONE_DEG:
        desired = right_key
    else:
        desired = None

    if desired != current_key_held:
        # release whatever was held, press the new one
        keyboard.release(left_key)
        keyboard.release(right_key)
        if desired is not None:
            keyboard.press(desired)
        current_key_held = desired


def draw_hud_panel(frame, status_text, output_mode, calibrated):
    """Draw a beautiful semi-transparent HUD panel on the top-left."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 320, 105
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (30, 30, 30), -1)
    # Add border
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (80, 80, 80), 1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw text labels
    cv2.putText(frame, status_text, (25, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Output Mode : {output_mode.upper()}", (25, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    status_color = (0, 255, 0) if calibrated else (0, 0, 255)
    calib_str = "YES" if calibrated else "NO (Press 'C')"
    cv2.putText(frame, "Calibrated  : ", (25, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, calib_str, (145, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)


def draw_virtual_wheel(frame, angle_deg, calibrated):
    """Draw a circular wheel overlay that rotates with the calculated angle."""
    h, w = frame.shape[:2]
    cx, cy = w - 110, 110
    radius = 70

    color = (0, 255, 0) if calibrated else (0, 0, 255)  # Green vs Red
    
    # Outer ring background (subtle grey ring)
    cv2.circle(frame, (cx, cy), radius, (60, 60, 60), 2)
    # Glow effect
    cv2.circle(frame, (cx, cy), radius, color, 6)
    cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2)

    # Spokes rotating with angle_deg
    rad = math.radians(angle_deg - 90)
    for spoke_offset in (0, 120, 240):
        a = rad + math.radians(spoke_offset)
        x2 = int(cx + radius * math.cos(a))
        y2 = int(cy + radius * math.sin(a))
        cv2.line(frame, (cx, cy), (x2, y2), color, 2)
        cv2.circle(frame, (x2, y2), 4, (255, 255, 255), -1)

    # Center hub
    cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)
    
    # Text overlay
    lbl_w, lbl_h = 140, 30
    lbl_x, lbl_y = cx - 70, cy + radius + 15
    overlay = frame.copy()
    cv2.rectangle(overlay, (lbl_x, lbl_y), (lbl_x + lbl_w, lbl_y + lbl_h), (30, 30, 30), -1)
    cv2.rectangle(overlay, (lbl_x, lbl_y), (lbl_x + lbl_w, lbl_y + lbl_h), (80, 80, 80), 1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, f"{angle_deg:+.1f} deg", (lbl_x + 15, lbl_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# ---------------------------------------------------------------------------
# 2D Arcade Racing Game Logic
# ---------------------------------------------------------------------------
def reset_game():
    global game_active, score, speed, player_x, obstacles, last_spawn_time
    game_active = True
    score = 0
    speed = 5
    player_x = 320
    obstacles = []
    last_spawn_time = time.time()


def draw_car(canvas, cx, cy, color):
    """Draw a top-down pixel-art style sports car."""
    w, h = 40, 70
    x1, y1 = cx - w//2, cy
    x2, y2 = cx + w//2, cy + h
    
    # Main body
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)
    
    # Roof/Windshield
    cv2.rectangle(canvas, (x1 + 6, y1 + 20), (x2 - 6, y1 + 45), (200, 200, 200), -1)
    # Hood lines
    cv2.line(canvas, (x1 + 8, y1), (x1 + 8, y1 + 15), (255, 255, 255), 1)
    cv2.line(canvas, (x2 - 8, y1), (x2 - 8, y1 + 15), (255, 255, 255), 1)
    
    # Wheels
    wheel_w, wheel_h = 8, 15
    cv2.rectangle(canvas, (x1 - wheel_w, y1 + 10), (x1, y1 + 10 + wheel_h), (30, 30, 30), -1)
    cv2.rectangle(canvas, (x2, y1 + 10), (x2 + wheel_w, y1 + 10 + wheel_h), (30, 30, 30), -1)
    cv2.rectangle(canvas, (x1 - wheel_w, y2 - 25), (x1, y2 - 25 + wheel_h), (30, 30, 30), -1)
    cv2.rectangle(canvas, (x2, y2 - 25), (x2 + wheel_w, y2 - 25 + wheel_h), (30, 30, 30), -1)


def draw_game_canvas():
    global road_scroll, score, high_score, speed, player_x, obstacles, last_spawn_time, game_active

    # Create dark empty BGR canvas
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Fill grass (Dark Green)
    canvas[:, :120] = (34, 139, 34)
    canvas[:, 520:] = (34, 139, 34)

    # 2. Fill road (Grey)
    canvas[:, 120:520] = (60, 60, 60)

    # 3. Draw yellow boundaries
    cv2.line(canvas, (120, 0), (120, 480), (0, 255, 255), 3)
    cv2.line(canvas, (520, 0), (520, 480), (0, 255, 255), 3)

    if game_active:
        # Update road scroll
        road_scroll = (road_scroll + speed) % 80

        # Spawn obstacles
        if len(obstacles) < 3 and (time.time() - last_spawn_time > 1.8):
            ob_x = random.randint(145, 495)
            ob_color = random.choice([
                (0, 0, 255),    # Red
                (0, 255, 255),  # Yellow
                (255, 0, 255),  # Pink
                (0, 165, 255)   # Orange
            ])
            obstacles.append({
                "x": ob_x,
                "y": -80,
                "color": ob_color,
                "speed_offset": random.uniform(-1.0, 1.5)
            })
            last_spawn_time = time.time()

        # Update obstacles
        active_obstacles = []
        for ob in obstacles:
            ob["y"] += int(speed + ob["speed_offset"])
            if ob["y"] < 480:
                active_obstacles.append(ob)
            else:
                score += 1
                if score > high_score:
                    high_score = score
                if score % 5 == 0:
                    speed = min(15, speed + 1)
        obstacles = active_obstacles

        # Check collision
        for ob in obstacles:
            if abs(player_x - ob["x"]) < 38 and abs(player_y - ob["y"]) < 65:
                game_active = False
                release_all_keys()

    # Draw road dashed lines
    for i in range(-1, 8):
        y_pos = int((i * 80 + road_scroll) % 480)
        cv2.line(canvas, (320, y_pos), (320, y_pos + 40), (255, 255, 255), 3)

    # Draw obstacles
    for ob in obstacles:
        draw_car(canvas, ob["x"], ob["y"], ob["color"])

    # Draw player car
    draw_car(canvas, int(player_x), player_y, (255, 0, 0))  # Blue player car

    # Draw HUD glass panel
    g_panel_w, g_panel_h = 240, 75
    g_overlay = canvas.copy()
    cv2.rectangle(g_overlay, (10, 10), (10 + g_panel_w, 10 + g_panel_h), (30, 30, 30), -1)
    cv2.rectangle(g_overlay, (10, 10), (10 + g_panel_w, 10 + g_panel_h), (80, 80, 80), 1)
    cv2.addWeighted(g_overlay, 0.6, canvas, 0.4, 0, canvas)

    cv2.putText(canvas, f"SCORE:      {score}", (25, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2)
    cv2.putText(canvas, f"HIGH SCORE: {high_score}", (25, 52), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)
    cv2.putText(canvas, f"SPEED:      {speed * 10} km/h", (25, 72), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 1)

    # Game Over screen overlay
    if not game_active:
        go_overlay = canvas.copy()
        cv2.rectangle(go_overlay, (0, 0), (640, 480), (10, 10, 10), -1)
        cv2.addWeighted(go_overlay, 0.75, canvas, 0.25, 0, canvas)
        
        cv2.putText(canvas, "GAME OVER", (170, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 255), 3)
        cv2.putText(canvas, f"Final Score: {score}", (240, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "Press 'R' to Restart", (210, 300), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
        cv2.putText(canvas, "Press 'Q' to Quit", (225, 330), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)

    return canvas


def main():
    global center_offset_deg, smoothed_angle, output_mode, player_x

    download_model_if_needed()

    # Setup detector
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONF,
        min_hand_presence_confidence=MIN_HAND_TRACKING_CONF
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Try to find a working camera index and backend
    cap = None
    print("[INFO] Scanning for a working webcam...")
    for index in (0, 1, 2, 3):
        # 1. Try Default backend
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"[INFO] Successfully opened camera index {index} using Default backend.")
                break
            cap.release()

        # 2. Try DirectShow backend
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"[INFO] Successfully opened camera index {index} using DSHOW backend.")
                break
            cap.release()

        # 3. Try Media Foundation backend
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"[INFO] Successfully opened camera index {index} using MSMF backend.")
                break
            cap.release()
            
        cap = None

    if cap is None:
        print("[ERROR] Could not open any working webcam (tried indices 0-3).")
        print("[ERROR] Please check:")
        print("        1) That your camera is connected and powered on.")
        print("        2) Windows Privacy Settings -> Camera (make sure 'Let desktop apps access your camera' is ON).")
        print("        3) That no other app (like Zoom, Teams, Skype, Chrome) is using the camera.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened: {w}x{h} @ ~{fps:.0f} FPS")
    print("[INFO] Controls: 'c' = calibrate center, 'm' = toggle keys/arrows, 'r' = restart game, 'q' = quit")

    calibrated = False
    window_name = "Virtual Steering Wheel"
    reset_game()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural steering feel
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe image and process
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)

        raw_angle = None
        detected_hands_count = len(results.hand_landmarks) if results.hand_landmarks else 0

        # Draw landmarks for whatever hands are detected
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                draw_hand_landmarks(frame, hand_landmarks, frame_w, frame_h)

        if detected_hands_count == 2:
            centers = []
            for hand_landmarks in results.hand_landmarks:
                centers.append(hand_center(hand_landmarks, frame_w, frame_h))

            # sort so left hand (smaller x) is first, right hand second
            centers.sort(key=lambda p: p[0])
            (x1, y1), (x2, y2) = centers

            # angle of the line between the two hands, relative to horizontal
            dx = x2 - x1
            dy = y2 - y1
            raw_angle = math.degrees(math.atan2(dy, dx))

            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 3)
            cv2.circle(frame, (int(x1), int(y1)), 10, (0, 255, 255), -1)
            cv2.circle(frame, (int(x2), int(y2)), 10, (0, 255, 255), -1)

        if detected_hands_count == 2 and raw_angle is not None:
            calibrated_angle = raw_angle - center_offset_deg
            smoothed_angle = (SMOOTHING * calibrated_angle) + (1 - SMOOTHING) * smoothed_angle
            display_angle = max(-90, min(90, smoothed_angle))

            # clamp/scale into +-100% steering feel based on FULL_LEFT/RIGHT thresholds
            send_steering_output(display_angle)
            status_text = "Tracking OK"

            # LINK TO GAME CAR POSITION
            if game_active:
                steer_ratio = display_angle / FULL_RIGHT_DEG
                steer_ratio = max(-1.0, min(1.0, steer_ratio))
                target_x = 320 + steer_ratio * 180
                player_x = player_x + 0.3 * (target_x - player_x)
        elif detected_hands_count == 1:
            smoothed_angle *= 0.9  # decay toward 0 if hands lost
            release_all_keys()
            status_text = "Only 1 hand detected (need 2)"
            if game_active:
                player_x = player_x + 0.1 * (320 - player_x)
        else:
            smoothed_angle *= 0.9  # decay toward 0 if hands lost
            release_all_keys()
            status_text = "Show BOTH hands to the camera"
            if game_active:
                player_x = player_x + 0.1 * (320 - player_x)

        # Draw overlays on webcam frame
        draw_virtual_wheel(frame, smoothed_angle, calibrated)
        draw_hud_panel(frame, status_text, output_mode, calibrated)

        if not KEYBOARD_AVAILABLE:
            cv2.putText(frame, "Key output DISABLED (see terminal warning)",
                        (20, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255), 2)

        # Draw 2D game canvas
        game_canvas = draw_game_canvas()

        # Combine webcam frame and game canvas side-by-side
        display_canvas = np.hstack((frame, game_canvas))

        cv2.imshow(window_name, display_canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c') and raw_angle is not None:
            center_offset_deg = raw_angle
            smoothed_angle = 0.0
            calibrated = True
            print(f"[INFO] Calibrated. Center offset set to {center_offset_deg:.1f} deg")
        elif key == ord('m'):
            output_mode = "arrows" if output_mode == "keys" else "keys"
            release_all_keys()
            print(f"[INFO] Output mode switched to: {output_mode}")
        elif key == ord('r'):
            reset_game()
            print("[INFO] Game restarted.")

    release_all_keys()
    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


