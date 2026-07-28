"""
Proof-of-concept: track one hand via webcam (MediaPipe) and mirror the
21 landmark points onto a simple wireframe "rig" (VPython spheres + cylinders).

Controls:
  - Show one hand to the webcam.
  - Press 'q' in the webcam window to quit.
"""

import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from vpython import canvas, sphere, cylinder, vector, rate, color

MODEL_PATH = "hand_landmarker.task"

# Bone connections between the 21 MediaPipe hand landmarks.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]

SCALE = 20  # world units across the normalized [0,1] landmark range


def build_rig():
    scene = canvas(title="Hand Rig", width=800, height=600, background=color.black)
    joints = [sphere(canvas=scene, pos=vector(0, 0, 0), radius=0.4, color=color.orange)
              for _ in range(21)]
    bones = [cylinder(canvas=scene, pos=vector(0, 0, 0), axis=vector(0, 0, 0),
                       radius=0.15, color=color.white)
              for _ in HAND_CONNECTIONS]
    return joints, bones


def landmark_to_vector(landmark):
    # MediaPipe gives normalized x,y in [0,1] (origin top-left) and a
    # rough relative z. Flip y/z so the rig moves the way you'd expect
    # when you move your hand up or toward the camera.
    x = (landmark.x - 0.5) * SCALE
    y = -(landmark.y - 0.5) * SCALE
    z = -landmark.z * SCALE
    return vector(x, y, z)


def update_rig(joints, bones, landmarks):
    positions = [landmark_to_vector(lm) for lm in landmarks]
    for joint, pos in zip(joints, positions):
        joint.pos = pos
    for bone, (i, j) in zip(bones, HAND_CONNECTIONS):
        bone.pos = positions[i]
        bone.axis = positions[j] - positions[i]


def draw_landmarks_on_frame(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for i, j in HAND_CONNECTIONS:
        cv2.line(frame, points[i], points[j], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
    )
    landmarker = HandLandmarker.create_from_options(options)

    joints, bones = build_rig()

    start_time = time.time()

    try:
        while True:
            rate(30)  # let VPython process/render this frame

            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                update_rig(joints, bones, landmarks)
                draw_landmarks_on_frame(frame, landmarks)

            cv2.imshow("Webcam (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
