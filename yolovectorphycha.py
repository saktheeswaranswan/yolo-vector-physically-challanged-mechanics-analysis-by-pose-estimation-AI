import os
# ── Suppress TF/MP logs ─────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.get_absl_handler().python_handler.stream = open(os.devnull, 'w')

import cv2
import numpy as np
import mediapipe as mp
import math

# ── Helper Functions ────────────────────────────────────────────────────
def rotate_vector(vec, theta):
    x, y = vec
    return np.array([x*math.cos(theta)-y*math.sin(theta),
                     x*math.sin(theta)+y*math.cos(theta)])

def calculate_angle(a, b, c):
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def draw_arc(img, center, angle, radius=30):
    axes = (radius, int(radius*0.6))
    cv2.ellipse(img, tuple(center.astype(int)), axes, 0, 0, angle, (0,255,255), 2)  # arc :contentReference[oaicite:4]{index=4}

# ── Setup Video Capture & Writer ───────────────────────────────────────
cap = cv2.VideoCapture('yolovector.mp4')
ret, tmp = cap.read()
h, w = tmp.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4 :contentReference[oaicite:5]{index=5}
writer = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))

# ── MediaPipe Holistic Setup ───────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

angle_deg, angle_step = 0, 15

with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # flip & process
        rgb = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        out = cv2.flip(frame,1)

        # draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(out, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(out, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # if we have pose+hand, compute fingertip vector
        if results.pose_landmarks and results.right_hand_landmarks:
            L = results.pose_landmarks.landmark
            H = results.right_hand_landmarks.landmark

            # get wrist & tip
            wrist = np.array([L[mp_holistic.PoseLandmark.RIGHT_WRIST].x * w,
                              L[mp_holistic.PoseLandmark.RIGHT_WRIST].y * h])
            tip   = np.array([H[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * w,
                              H[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * h])

            # control vector at fingertip
            vec = tip - wrist

            # rotate on 'r'
            if cv2.waitKey(1) & 0xFF == ord('r'):
                angle_deg = (angle_deg + angle_step) % 360
            theta = -math.radians(angle_deg)
            vec_rot = rotate_vector(vec, theta)
            rot_tip = wrist + vec_rot

            # draw original & rotated
            cv2.arrowedLine(out, tuple(wrist.astype(int)), tuple(rot_tip.astype(int)), (0,0,255), 2)  # arrow :contentReference[oaicite:6]{index=6}

            # compute mag & dir
            mag = np.linalg.norm(vec_rot)  # magnitude :contentReference[oaicite:7]{index=7}
            dir_deg = (np.degrees(np.arctan2(vec_rot[1], vec_rot[0])) + 360) % 360  # direction :contentReference[oaicite:8]{index=8}

            # annotate
            text = f"M={mag:.1f}, θ={dir_deg:.1f}°"
            cv2.putText(out, text, tuple((rot_tip + np.array([10,-10])).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # draw angle arc at wrist for reference
            draw_arc(out, wrist, angle_deg)

        # write & show
        writer.write(out)  
        cv2.imshow('Control Vector & Rotation', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
writer.release()
cv2.destroyAllWindows()

