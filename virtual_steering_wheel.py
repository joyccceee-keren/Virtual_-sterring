"""
F1 Telemetry & Racing Simulator
-------------------------------
Left side (640x480): Camera feed with hand skeleton tracking and F1 yoke.
Right side (640x480): A scrolling 2D arcade car game controlled by steering.

Controls:
  c   - calibrate (set current hand angle as center/straight)
  m   - toggle keyboard output mode (WASD vs Arrows)
  r   - restart the race
  q   - quit (or close the window)

Install dependencies first:
  pip install opencv-python mediapipe keyboard numpy pygame --break-system-packages
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
import pygame
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
# Hand Connections
# ---------------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17), (0, 17)
]

# ---------------------------------------------------------------------------
# Calibration & State
# ---------------------------------------------------------------------------
center_offset_deg = 0.0      # calibration offset
smoothed_angle = 0.0         # filtered wheel angle relative to center
output_mode = "keys"         # "keys" (a/d) or "arrows" (left/right)
current_key_held = None      # tracks which key is currently pressed down

# ---------------------------------------------------------------------------
# Simulator Physics & Timing State
# ---------------------------------------------------------------------------
game_active = True
speed = 0.0                  # vehicle speed in km/h (0 to 330)
target_speed = 0.0
rpm = 4000.0                 # engine RPM (4000 to 13500)
player_x = 320.0
player_y = 380.0
obstacles = []
last_spawn_time = 0.0
road_scroll = 0.0

# Lap Timing Telemetry
lap_start_time = None
current_lap_time = 0.0
best_lap_time = None
last_lap_time = None
actual_distance_this_lap = 0.0
LAP_TARGET_DISTANCE = 1500.0  # 1.5 km lap length
TARGET_PACE_SPEED = 210.0     # Reference pace in km/h for Delta calculation


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

    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], (200, 200, 200), 2)

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
        keyboard.release(left_key)
        keyboard.release(right_key)
        if desired is not None:
            keyboard.press(desired)
        current_key_held = desired


# ---------------------------------------------------------------------------
# F1 Steering Wheel Overlay (Rotates with smoothed_angle)
# ---------------------------------------------------------------------------
def rotate_point(pt, center, angle_deg):
    """Rotate a point (x, y) around center by angle_deg."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    rx = center[0] + dx * cos_a - dy * sin_a
    ry = center[1] + dx * sin_a + dy * cos_a
    return (int(rx), int(ry))


def draw_f1_yoke(frame, cx, cy, angle_deg, current_rpm):
    """Draw a rotated F1 steering yoke with RPM LED rev lights."""
    # Base yoke shapes relative to center
    body_rel = [(-45, -20), (45, -20), (40, 25), (-40, 25)]
    left_grip_rel = [(-60, -40), (-42, -40), (-38, 35), (-56, 35)]
    right_grip_rel = [(42, -40), (60, -40), (56, 35), (38, 35)]
    screen_rel = [(-22, -10), (22, -10), (20, 12), (-20, 12)]

    # Rotate coordinates
    body_pts = np.array([rotate_point((cx + x, cy + y), (cx, cy), angle_deg) for x, y in body_rel])
    left_pts = np.array([rotate_point((cx + x, cy + y), (cx, cy), angle_deg) for x, y in left_grip_rel])
    right_pts = np.array([rotate_point((cx + x, cy + y), (cx, cy), angle_deg) for x, y in right_grip_rel])
    screen_pts = np.array([rotate_point((cx + x, cy + y), (cx, cy), angle_deg) for x, y in screen_rel])

    # 1. Draw Grips (dark grey/black carbon color)
    cv2.fillPoly(frame, [left_pts], (25, 25, 25))
    cv2.fillPoly(frame, [right_pts], (25, 25, 25))
    cv2.polylines(frame, [left_pts], True, (60, 60, 60), 1)
    cv2.polylines(frame, [right_pts], True, (60, 60, 60), 1)

    # 2. Draw Yoke Body (grey)
    cv2.fillPoly(frame, [body_pts], (50, 50, 50))
    cv2.polylines(frame, [body_pts], True, (120, 120, 120), 1)

    # 3. Draw Buttons (neon blue, red, yellow, green)
    button_locs = [(-25, -28), (25, -28), (-50, -5), (50, -5)]
    button_colors = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255)]  # BGR
    for loc, col in zip(button_locs, button_colors):
        btn_pt = rotate_point((cx + loc[0], cy + loc[1]), (cx, cy), angle_deg)
        cv2.circle(frame, btn_pt, 4, col, -1)

    # 4. Draw Center Telemetry Screen (dark dashboard screen)
    cv2.fillPoly(frame, [screen_pts], (15, 15, 15))
    cv2.polylines(frame, [screen_pts], True, (0, 255, 255), 1)

    # 5. Draw 15 LED RPM rev lights across the top of the yoke
    # LEDs relative center positions
    led_rel = [(-38 + i * 5.4, -30) for i in range(15)]
    # Determine how many LEDs to light up (RPM from 4000 to 13500)
    num_leds = int(max(0, min(15, (current_rpm - 4000) / 9000 * 15))) if current_rpm > 4000 else 0

    for i, rel_pos in enumerate(led_rel):
        led_pt = rotate_point((cx + rel_pos[0], cy + rel_pos[1]), (cx, cy), angle_deg)
        
        # Color mapping (5 Green -> 5 Red -> 5 Blue)
        if i < 5:
            col = (0, 220, 0) if i < num_leds else (0, 50, 0)       # Green
        elif i < 10:
            col = (0, 0, 220) if i < num_leds else (0, 0, 50)       # Red
        else:
            col = (220, 0, 150) if i < num_leds else (50, 0, 40)    # Blue/Purple
            
        cv2.circle(frame, led_pt, 3, col, -1)
        cv2.circle(frame, led_pt, 3, (120, 120, 120), 1)


# ---------------------------------------------------------------------------
# F1 Telemetry HUD Overlay (Left side Carbon Fiber Panel)
# ---------------------------------------------------------------------------
def draw_carbon_panel(frame, x, y, w, h):
    """Draw a carbon-fiber textured HUD background panel."""
    # Draw dark panel backing
    cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 20, 20), -1)

    # Draw carbon fiber weave pattern (diagonal stripes)
    for offset in range(-h, w, 4):
        x1 = max(x, x + offset)
        y1 = y + (x1 - (x + offset))
        x2 = min(x + w, x + offset + h)
        y2 = y + (x2 - (x + offset))
        if x1 < x2:
            cv2.line(frame, (x1, y1), (x2, y2), (28, 28, 28), 1)

    # Panel border
    cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1)


def draw_hud_panel(frame, status_text, output_mode, calibrated, angle_deg, current_speed, current_rpm):
    """Draw the detailed F1 yoke carbon telemetry HUD panel."""
    h, w = frame.shape[:2]
    panel_x, panel_y = 15, 15
    panel_w, panel_h = 320, 115
    
    draw_carbon_panel(frame, panel_x, panel_y, panel_w, panel_h)

    # Telemetry text values
    cv2.putText(frame, "TELEMETRY OVERVIEW", (panel_x + 15, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
    
    cv2.putText(frame, f"STATUS: {status_text.upper()}", (panel_x + 15, panel_y + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, f"STEER ANGLE: {angle_deg:+.1f} DEG", (panel_x + 15, panel_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    calib_col = (0, 255, 0) if calibrated else (0, 0, 255)
    calib_val = "READY" if calibrated else "NOT CALIBRATED (PRESS 'C')"
    cv2.putText(frame, "CALIBRATION: ", (panel_x + 15, panel_y + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, calib_val, (panel_x + 115, panel_y + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, calib_col, 1)

    cv2.putText(frame, f"INPUT MODE: {output_mode.upper()}", (panel_x + 15, panel_y + 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)


# ---------------------------------------------------------------------------
# Pygame Racetrack & Car Graphics (Right Side)
# ---------------------------------------------------------------------------
def pygame_draw_f1_car(surface, cx, cy, color, is_player=False):
    """Draw a detailed top-down open-wheel F1 car silhouette."""
    w, h = 44, 80
    x1 = cx - w // 2
    y1 = cy

    # 1. Exposed Tires (Front and Rear, black)
    tire_w, tire_h = 10, 16
    # Front tires
    pygame.draw.rect(surface, (15, 15, 15), (x1 - 4, y1 + 12, tire_w, tire_h))
    pygame.draw.rect(surface, (15, 15, 15), (x1 + w - 6, y1 + 12, tire_w, tire_h))
    # Rear tires (wider)
    rtire_w, rtire_h = 12, 18
    pygame.draw.rect(surface, (15, 15, 15), (x1 - 6, y1 + h - 25, rtire_w, rtire_h))
    pygame.draw.rect(surface, (15, 15, 15), (x1 + w - 6, y1 + h - 25, rtire_w, rtire_h))

    # Tire metal rims (silver centers)
    pygame.draw.circle(surface, (80, 80, 80), (int(x1 - 4 + tire_w/2), int(y1 + 12 + tire_h/2)), 3)
    pygame.draw.circle(surface, (80, 80, 80), (int(x1 + w - 6 + tire_w/2), int(y1 + 12 + tire_h/2)), 3)
    pygame.draw.circle(surface, (80, 80, 80), (int(x1 - 6 + rtire_w/2), int(y1 + h - 25 + rtire_h/2)), 4)
    pygame.draw.circle(surface, (80, 80, 80), (int(x1 + w - 6 + rtire_w/2), int(y1 + h - 25 + rtire_h/2)), 4)

    # 2. Main F1 Aerodynamic Pod Body
    # Front nose cone (very narrow)
    pygame.draw.rect(surface, color, (cx - 5, y1 + 8, 10, 20))
    # Tapered sidepods
    pygame.draw.polygon(surface, color, [
        (cx - 5, y1 + 28), (cx + 5, y1 + 28),
        (cx + 15, y1 + 45), (cx + 12, y1 + 65),
        (cx - 12, y1 + 65), (cx - 15, y1 + 45)
    ])

    # 3. Front Wing (horizontal wing at the front nose tip)
    pygame.draw.rect(surface, (30, 30, 30), (x1 - 10, y1 + 2, w + 20, 6))
    pygame.draw.rect(surface, color, (x1 - 10, y1, 3, 10))
    pygame.draw.rect(surface, color, (x1 + w + 7, y1, 3, 10))

    # 4. Rear Wing (wide rear wing)
    pygame.draw.rect(surface, (30, 30, 30), (x1 - 12, y1 + h - 6, w + 24, 6))
    pygame.draw.rect(surface, color, (x1 - 12, y1 + h - 10, 3, 12))
    pygame.draw.rect(surface, color, (x1 + w + 9, y1 + h - 10, 3, 12))

    # 5. Halo Protective Loop & Cockpit
    pygame.draw.ellipse(surface, (10, 10, 10), (cx - 7, y1 + 32, 14, 18))
    # Driver helmet (Yellow for player, Orange for obstacles)
    helmet_color = (255, 255, 0) if is_player else (255, 100, 0)
    pygame.draw.circle(surface, helmet_color, (cx, y1 + 41), 5)
    # VISOR
    pygame.draw.rect(surface, (0, 0, 0), (cx - 4, y1 + 38, 8, 2))
    # Halo loop structures
    pygame.draw.polygon(surface, (40, 40, 40), [
        (cx, y1 + 30), (cx + 8, y1 + 34), (cx + 8, y1 + 48),
        (cx + 6, y1 + 48), (cx + 6, y1 + 36), (cx, y1 + 32),
        (cx - 6, y1 + 36), (cx - 6, y1 + 48), (cx - 8, y1 + 48),
        (cx - 8, y1 + 34)
    ])

    # 6. Racing Number
    pygame.draw.rect(surface, (255, 255, 255), (cx - 3, y1 + 15, 6, 8))
    num_font = pygame.font.Font(None, 12)
    num_txt = num_font.render("1" if is_player else "4", True, (0, 0, 0))
    surface.blit(num_txt, (cx - 2, y1 + 14))

    # 7. Headlight glow (for player car only)
    if is_player:
        glow_surf = pygame.Surface((640, 480), pygame.SRCALPHA)
        # Left head beam
        pygame.draw.polygon(glow_surf, (255, 255, 150, 20), [
            (cx - 15, y1), (cx - 70, y1 - 120), (cx + 10, y1 - 120)
        ])
        # Right head beam
        pygame.draw.polygon(glow_surf, (255, 255, 150, 20), [
            (cx + 15, y1), (cx - 10, y1 - 120), (cx + 70, y1 - 120)
        ])
        surface.blit(glow_surf, (0, 0))


def draw_game_surface(game_surf, font_large, font_medium, font_small, dt):
    """Draw the Pygame 2D Racing Game onto a local Pygame surface."""
    global road_scroll, speed, player_x, obstacles, last_spawn_time, game_active
    global lap_start_time, current_lap_time, best_lap_time, last_lap_time, actual_distance_this_lap

    # 1. Fill grass (dark green)
    game_surf.fill((34, 110, 34))

    # 2. Fill asphalt road (dark grey)
    pygame.draw.rect(game_surf, (50, 50, 50), (120, 0, 400, 480))

    # 3. Draw Red & White striped rumble strips (kerbs)
    for i in range(-2, 14):
        y_pos = int((i * 40 + road_scroll) % 480)
        col = (255, 255, 255) if (i + int(road_scroll // 40)) % 2 == 0 else (200, 0, 0)
        # Left kerb
        pygame.draw.rect(game_surf, col, (112, y_pos, 8, 40))
        # Right kerb
        pygame.draw.rect(game_surf, col, (520, y_pos, 8, 40))

    # 4. Draw center lane dashed line
    for i in range(-1, 8):
        y_pos = int((i * 80 + road_scroll) % 480)
        pygame.draw.rect(game_surf, (220, 220, 220), (318, y_pos, 4, 40))

    # 5. Draw asphalt speed line overlays
    for i in range(3):
        y_pos = int((i * 160 + road_scroll * 1.4) % 480)
        pygame.draw.line(game_surf, (44, 44, 44), (150 + i * 80, y_pos), (150 + i * 80, y_pos + 50), 2)
        pygame.draw.line(game_surf, (44, 44, 44), (380 + i * 60, y_pos), (380 + i * 60, y_pos + 30), 2)

    # 6. Obstacle Spawning and Updates
    if game_active:
        # Increment road scroll
        road_scroll = (road_scroll + speed * 0.15) % 80

        # Increment distance & time
        actual_distance_this_lap += (speed / 3.6) * dt
        current_lap_time = time.time() - lap_start_time

        # Lap completion logic
        if actual_distance_this_lap >= LAP_TARGET_DISTANCE:
            last_lap_time = current_lap_time
            if best_lap_time is None or last_lap_time < best_lap_time:
                best_lap_time = last_lap_time
            # reset lap variables
            actual_distance_this_lap = 0.0
            lap_start_time = time.time()

        # Spawn obstacle cars
        if len(obstacles) < 3 and (time.time() - last_spawn_time > 1.8):
            ob_x = random.randint(145, 495)
            ob_col = random.choice([
                (255, 30, 30),     # Red
                (255, 220, 0),    # Yellow
                (255, 50, 150),   # Pink
                (255, 120, 0)     # Orange
            ])
            obstacles.append({
                "x": ob_x,
                "y": -80,
                "color": ob_col,
                "speed_offset": random.uniform(-5.0, 25.0)
            })
            last_spawn_time = time.time()

        # Move obstacles (speed is relative to player speed)
        active_obstacles = []
        for ob in obstacles:
            ob_scroll_speed = (speed - ob["speed_offset"]) * 0.15
            ob["y"] += max(2, int(ob_scroll_speed))
            
            if ob["y"] < 480:
                active_obstacles.append(ob)
        obstacles = active_obstacles

        # Collision detection (Exposed tire hitboxes make it challenging!)
        for ob in obstacles:
            if abs(player_x - ob["x"]) < 38 and abs(player_y - ob["y"]) < 65:
                game_active = False
                release_all_keys()
                speed = 0.0

    # Draw Obstacles
    for ob in obstacles:
        pygame_draw_f1_car(game_surf, ob["x"], ob["y"], ob["color"])

    # Draw Player F1 Car (Neon Blue/Cyan)
    pygame_draw_f1_car(game_surf, int(player_x), int(player_y), (0, 180, 255), is_player=True)

    # 7. Draw Telemetry Dashboard Overlay (Glass Box)
    hud_overlay = pygame.Surface((280, 120), pygame.SRCALPHA)
    pygame.draw.rect(hud_overlay, (15, 15, 15, 210), (0, 0, 280, 120))
    pygame.draw.rect(hud_overlay, (90, 90, 90, 255), (0, 0, 280, 120), 1)
    game_surf.blit(hud_overlay, (10, 10))

    # Dynamic Gear Indicator (Large Box)
    if speed <= 10.0:
        gear_char = "N"
    elif speed <= 45.0:
        gear_char = "1"
    elif speed <= 80.0:
        gear_char = "2"
    elif speed <= 120.0:
        gear_char = "3"
    elif speed <= 160.0:
        gear_char = "4"
    elif speed <= 200.0:
        gear_char = "5"
    elif speed <= 240.0:
        gear_char = "6"
    elif speed <= 285.0:
        gear_char = "7"
    else:
        gear_char = "8"

    pygame.draw.rect(game_surf, (0, 255, 255), (20, 20, 52, 52), 2)
    gear_txt = font_large.render(gear_char, True, (0, 255, 255))
    game_surf.blit(gear_txt, (35, 28))
    gear_lbl = font_small.render("GEAR", True, (200, 200, 200))
    game_surf.blit(gear_lbl, (30, 75))

    # Animated RPM Bar (Visual Tachometer)
    if gear_char == "N":
        rpm = 4000.0 + (speed / 10.0) * 2000.0
    else:
        gear_bounds = {
            "1": (10.0, 45.0), "2": (45.0, 80.0), "3": (80.0, 120.0),
            "4": (120.0, 160.0), "5": (160.0, 200.0), "6": (200.0, 240.0),
            "7": (240.0, 285.0), "8": (285.0, 340.0)
        }
        g_min, g_max = gear_bounds[gear_char]
        ratio = (speed - g_min) / (g_max - g_min)
        ratio = max(0.0, min(1.0, ratio))
        rpm = 5000.0 + ratio * 8200.0

    pygame.draw.rect(game_surf, (80, 80, 80), (85, 20, 190, 14), 1)
    rpm_fill_width = int(max(0, min(188, (rpm - 4000) / 9500 * 188)))
    
    rpm_ratio = (rpm - 4000) / 9500
    if rpm_ratio < 0.6:
        rpm_col = (0, 255, 0)
    elif rpm_ratio < 0.85:
        rpm_col = (255, 128, 0)
    else:
        rpm_col = (255, 0, 128) if int(time.time() * 15) % 2 == 0 else (0, 255, 255)
        
    pygame.draw.rect(game_surf, rpm_col, (86, 21, rpm_fill_width, 12))
    
    rpm_txt = font_small.render(f"RPM: {int(rpm)}", True, (255, 255, 255))
    game_surf.blit(rpm_txt, (85, 38))

    # Speed text
    speed_txt = font_medium.render(f"{int(speed)} KM/H", True, (255, 255, 255))
    game_surf.blit(speed_txt, (175, 36))

    # Live lap time display
    m, s = divmod(current_lap_time, 60)
    lap_str = f"LAP: {int(m):02d}:{s:06.3f}"
    lap_txt = font_small.render(lap_str, True, (255, 255, 255))
    game_surf.blit(lap_txt, (85, 58))

    # Best lap time display
    if best_lap_time is not None:
        bm, bs = divmod(best_lap_time, 60)
        best_str = f"BEST: {int(bm):02d}:{bs:06.3f}"
    else:
        best_str = "BEST: --:--.---"
    best_txt = font_small.render(best_str, True, (200, 200, 200))
    game_surf.blit(best_txt, (85, 75))

    # F1 Telemetry Live Delta
    if game_active and current_lap_time > 0.5:
        target_dist = (TARGET_PACE_SPEED / 3.6) * current_lap_time
        delta = (target_dist - actual_distance_this_lap) / (TARGET_PACE_SPEED / 3.6)
        
        delta_sign = "+" if delta >= 0 else "-"
        delta_color = (255, 30, 30) if delta >= 0 else (0, 255, 0)
        delta_str = f"DELTA: {delta_sign}{abs(delta):.3f}s"
    else:
        delta_str = "DELTA: +0.000s"
        delta_color = (200, 200, 200)
    
    delta_txt = font_small.render(delta_str, True, delta_color)
    game_surf.blit(delta_txt, (85, 92))

    # DRS Badge Overlay (DRS is available above 250 km/h)
    if speed >= 250.0:
        drs_box = pygame.Surface((75, 22))
        drs_box.fill((0, 255, 0) if speed < 300.0 else (0, 255, 255))
        drs_text_col = (0, 0, 0)
        drs_str = "DRS ON" if speed >= 300.0 else "DRS AVAIL"
        
        if speed >= 300.0 and int(time.time() * 5) % 2 == 0:
            drs_box.fill((10, 10, 10))
            drs_text_col = (0, 255, 255)
            
        game_surf.blit(drs_box, (190, 75))
        drs_txt = font_small.render(drs_str, True, drs_text_col)
        game_surf.blit(drs_txt, (197, 79))

    # 8. Game Over Screen Overlay
    if not game_active:
        go_surf = pygame.Surface((640, 480), pygame.SRCALPHA)
        pygame.draw.rect(go_surf, (10, 10, 10, 210), (0, 0, 640, 480))
        game_surf.blit(go_surf, (0, 0))

        go_lbl = font_large.render("GAME OVER", True, (255, 30, 30))
        game_surf.blit(go_lbl, (210, 160))
        
        score_lbl = font_medium.render(f"Lap Completion: {int(actual_distance_this_lap / LAP_TARGET_DISTANCE * 100)}%", True, (255, 255, 255))
        game_surf.blit(score_lbl, (215, 220))
        
        restart_lbl = font_medium.render("Press 'R' to Restart", True, (0, 255, 0))
        game_surf.blit(restart_lbl, (225, 270))
        
        quit_lbl = font_small.render("Press 'Q' to Quit", True, (200, 200, 200))
        game_surf.blit(quit_lbl, (260, 310))


def main():
    global center_offset_deg, smoothed_angle, output_mode, player_x
    global game_active, speed, target_speed, rpm, obstacles, last_spawn_time

    download_model_if_needed()

    # Initialize Pygame Display
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((1280, 480))
    pygame.display.set_caption("F1 Virtual Steering Wheel & Telemetry Simulator")
    
    font_large = pygame.font.Font(None, 44)
    font_medium = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 18)
    
    pygame_clock = pygame.time.Clock()

    # Setup MediaPipe detector
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONF,
        min_hand_presence_confidence=MIN_HAND_TRACKING_CONF
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Webcam scanning loop
    cap = None
    print("[INFO] Scanning for a working webcam...")
    for index in (0, 1, 2, 3):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"[INFO] Successfully opened camera index {index} using Default backend.")
                break
            cap.release()

        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"[INFO] Successfully opened camera index {index} using DSHOW backend.")
                break
            cap.release()

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
        pygame.quit()
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened: {w}x{h} @ ~{fps:.0f} FPS")

    calibrated = False
    reset_game()

    # Sub-surface for F1 road game
    game_surface = pygame.Surface((640, 480))

    running = True
    while running:
        # Get frame-by-frame delta time (dt) in seconds
        dt = pygame_clock.tick(30) / 1000.0

        # Handle Pygame Input Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    reset_game()
                    print("[INFO] Game restarted.")
                elif event.key == pygame.K_m:
                    output_mode = "arrows" if output_mode == "keys" else "keys"
                    release_all_keys()
                    print(f"[INFO] Output mode switched to: {output_mode}")
                elif event.key == pygame.K_c:
                    # Trigger calibration inside logic
                    pass

        # OpenCV capture
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)  # mirror frame
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)

        raw_angle = None
        detected_hands_count = len(results.hand_landmarks) if results.hand_landmarks else 0

        # Draw hand skeletons
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                draw_hand_landmarks(frame, hand_landmarks, frame_w, frame_h)

        if detected_hands_count == 2:
            centers = []
            for hand_landmarks in results.hand_landmarks:
                centers.append(hand_center(hand_landmarks, frame_w, frame_h))

            centers.sort(key=lambda p: p[0])
            (x1, y1), (x2, y2) = centers

            dx = x2 - x1
            dy = y2 - y1
            raw_angle = math.degrees(math.atan2(dy, dx))

            # Track center connector line
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)
            cv2.circle(frame, (int(x1), int(y1)), 6, (0, 255, 255), -1)
            cv2.circle(frame, (int(x2), int(y2)), 6, (0, 255, 255), -1)

        # Trigger calibration check
        keys_pressed = pygame.key.get_pressed()
        if keys_pressed[pygame.K_c] and raw_angle is not None:
            center_offset_deg = raw_angle
            smoothed_angle = 0.0
            calibrated = True
            print(f"[INFO] Calibrated. Center offset set to {center_offset_deg:.1f} deg")

        # Steering control linkage
        if detected_hands_count == 2 and raw_angle is not None:
            calibrated_angle = raw_angle - center_offset_deg
            smoothed_angle = (SMOOTHING * calibrated_angle) + (1 - SMOOTHING) * smoothed_angle
            display_angle = max(-90, min(90, smoothed_angle))

            send_steering_output(display_angle)
            status_text = "Tracking OK"

            # F1 Acceleration: increase speed smoothly when hands are tracked
            if game_active:
                max_track_speed = 330.0 - abs(display_angle) * 1.2
                target_speed = max_track_speed
                speed = speed + 4.5 * (target_speed - speed) * dt

                # Move F1 car
                steer_ratio = display_angle / FULL_RIGHT_DEG
                steer_ratio = max(-1.0, min(1.0, steer_ratio))
                target_x = 320.0 + steer_ratio * 180.0
                player_x = player_x + 6.0 * (target_x - player_x) * dt
        else:
            smoothed_angle *= 0.9
            release_all_keys()
            
            # Engine braking: decelerate car quickly if hands are lost
            target_speed = 0.0
            speed = speed + 8.0 * (target_speed - speed) * dt
            
            if game_active:
                player_x = player_x + 3.0 * (320.0 - player_x) * dt

            if detected_hands_count == 1:
                status_text = "Only 1 hand detected (need 2)"
            else:
                status_text = "Show BOTH hands to the camera"

        # Draw overlays on BGR OpenCV frame
        draw_f1_yoke(frame, frame_w - 120, 120, smoothed_angle, rpm)
        draw_hud_panel(frame, status_text, output_mode, calibrated, smoothed_angle, speed, rpm)

        if not KEYBOARD_AVAILABLE:
            cv2.putText(frame, "Key output DISABLED (see terminal warning)",
                        (20, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255), 2)

        # Convert OpenCV BGR to Pygame RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_surface = pygame.image.frombuffer(rgb_frame.tobytes(), (640, 480), "RGB")

        # Draw the 2D Game Surface
        draw_game_surface(game_surface, font_large, font_medium, font_small, dt)

        # Blit panels onto Pygame main window
        screen.blit(camera_surface, (0, 0))
        screen.blit(game_surface, (640, 0))

        # Render display
        pygame.display.flip()

    release_all_keys()
    detector.close()
    cap.release()
    pygame.quit()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
