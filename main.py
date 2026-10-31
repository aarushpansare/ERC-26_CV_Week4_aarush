import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pygame

pygame.mixer.init()
AI_MODEL_FILE = 'hand_landmarker.task'
NOTE_FILES = {
    1: "1_Sa.mp3",
    2: "2_Re.mp3",
    3: "3_Ga.mp3",
    4: "4_Ma.mp3",
    5: "5_Pa.mp3",
    6: "6_Dha.mp3",
    7: "7_Ni.mp3",
    8: "8_Sa_upper.mp3"
}

audio_clips = {}
for digit_count, file_name in NOTE_FILES.items():
    full_path = "sounds_swaras/" + file_name
    try:
        audio_clips[digit_count] = pygame.mixer.Sound(full_path)
    except Exception as err:
        print(f"Failed to load track: {full_path}. Reason: {err}")

SKELETON_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb connections
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index connections
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle connections
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring connections
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky and Palm
]

def calculate_raised_digits(detected_hands_data):
    active_count = 0
    distal_joints = [8, 12, 16, 20]
    middle_joints = [6, 10, 14, 18]
    
    for hand_data in detected_hands_data:
        for tip_node, pip_node in zip(distal_joints, middle_joints):
            if hand_data[tip_node].y < hand_data[pip_node].y:
                active_count += 1                
        
        if abs(hand_data[4].x - hand_data[17].x) > abs(hand_data[3].x - hand_data[17].x):
            active_count += 1
            
    return active_count

def overlay_keypoints(image_canvas, hand_data):
    img_h, img_w, _ = image_canvas.shape
    pixel_coords = []
    
    for node in hand_data:
        pos_x = node.x if hasattr(node, 'x') else node[0]
        pos_y = node.y if hasattr(node, 'y') else node[1]
        pixel_coords.append((int(pos_x * img_w), int(pos_y * img_h)))
    
    for idx_start, idx_end in SKELETON_PAIRS:
        if idx_start < len(pixel_coords) and idx_end < len(pixel_coords):
            cv2.line(image_canvas, pixel_coords[idx_start], pixel_coords[idx_end], (0, 255, 0), 2)
            
    for x_val, y_val in pixel_coords:
        cv2.circle(image_canvas, (x_val, y_val), 4, (0, 0, 255), -1)

try:
    model_config = mp_python.BaseOptions(model_asset_path=AI_MODEL_FILE)
    vision_settings = mp_vision.HandLandmarkerOptions(
        base_options=model_config,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
    )
    gesture_detector = mp_vision.HandLandmarker.create_from_options(vision_settings)
except Exception as err:
    print(f"\nUnable to initialize model from '{AI_MODEL_FILE}'.")
    exit(1)

video_feed = cv2.VideoCapture(0)
previous_time_ms = -1
currently_playing_idx = -1

print("Music Player launched. Press 'q' to terminate.")

while video_feed.isOpened():
    read_success, cam_frame = video_feed.read()
    if not read_success or cam_frame is None:
        break

    cam_frame = cv2.flip(cam_frame, 1)
    color_rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
    color_rgb = np.ascontiguousarray(color_rgb, dtype=np.uint8)
    mp_wrapper = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_rgb)
    
    current_time_ms = int(time.monotonic() * 1000)
    if current_time_ms <= previous_time_ms:
        current_time_ms = previous_time_ms + 1
    previous_time_ms = current_time_ms

    inference_result = gesture_detector.detect_for_video(mp_wrapper, current_time_ms)
    detected_digits = 0
    
    if inference_result and inference_result.hand_landmarks:
        for hand_info in inference_result.hand_landmarks:
            overlay_keypoints(cam_frame, hand_info)
        detected_digits = calculate_raised_digits(inference_result.hand_landmarks)

    if detected_digits in [0, 9]:
        if currently_playing_idx != 0:
            pygame.mixer.stop()
            currently_playing_idx = 0
    elif detected_digits in audio_clips:
        if detected_digits != currently_playing_idx:
            pygame.mixer.stop() 
            audio_clips[detected_digits].play()
            currently_playing_idx = detected_digits

    display_msg = f"Fingers: {detected_digits} | Swara Note: {NOTE_FILES.get(detected_digits, 'Silence')}"
    cv2.putText(cam_frame, display_msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow('Gesture Music Interface', cam_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_feed.release()
cv2.destroyAllWindows()
gesture_detector.close()