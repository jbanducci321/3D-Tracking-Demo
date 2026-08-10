"""
Proof-of-concept: track a full body via webcam (MediaPipe Pose) and mirror
the 33 landmark points onto a simple wireframe "rig" (VPython spheres +
cylinders). Same approach as hand_demo.py, scaled up to the whole body.

Controls:
  - Stand in front of the webcam.
  - Press 'q' in the webcam window to quit.
"""

import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python.vision import PoseLandmarksConnections
from vpython import canvas, sphere, cylinder, vector, rate, color

MODEL_PATH = "pose_landmarker_lite.task"

# Official 33-landmark body skeleton connections, straight from MediaPipe.
POSE_CONNECTIONS = [(c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS]

# Physical size (in world units) the wider of the frame's two dimensions
# should map to. Landmark x/y are normalized [0,1] fractions of frame
# width/height respectively, so scaling by actual pixel dimensions (rather
# than a single flat SCALE like the hand demo) keeps body proportions
# correct instead of stretching/squishing on non-square webcam frames.
WORLD_SIZE = 20


def build_rig():
    scene = canvas(title="Body Rig", width=800, height=600, background=color.black)
    joints = [sphere(canvas=scene, pos=vector(0, 0, 0), radius=0.3, color=color.orange)
              for _ in range(33)]
    bones = [cylinder(canvas=scene, pos=vector(0, 0, 0), axis=vector(0, 0, 0),
                       radius=0.12, color=color.white)
             for _ in POSE_CONNECTIONS]
    return joints, bones


def landmark_to_vector(landmark, frame_w, frame_h):
    unit = WORLD_SIZE / max(frame_w, frame_h)
    x = (landmark.x - 0.5) * frame_w * unit
    y = -(landmark.y - 0.5) * frame_h * unit
    z = -landmark.z * frame_w * unit
    return vector(x, y, z)


def update_rig(joints, bones, landmarks, frame_w, frame_h):
    positions = [landmark_to_vector(lm, frame_w, frame_h) for lm in landmarks]
    for joint, pos in zip(joints, positions):
        joint.pos = pos
    for bone, (i, j) in zip(bones, POSE_CONNECTIONS):
        bone.pos = positions[i]
        bone.axis = positions[j] - positions[i]


def draw_landmarks_on_frame(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for i, j in POSE_CONNECTIONS:
        cv2.line(frame, points[i], points[j], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    joints, bones = build_rig()

    start_time = time.time()

    try:
        while True:
            rate(30)  # let VPython process/render this frame

            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                update_rig(joints, bones, landmarks, frame_w, frame_h)
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
