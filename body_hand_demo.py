"""
Proof-of-concept: track a full body AND up to two hands via webcam, using
two independent MediaPipe models (Pose + Hand Landmarker) fed the same
frame, and mirror both onto shared VPython wireframe "rigs".

Body and hand landmarks are both normalized (x, y) fractions of the same
webcam frame, so they share one coordinate space already -- we scale both
the same way (see landmark_to_vector). But the two models still disagree
slightly on exactly where each wrist is (independent detectors, and their
z-depth is relative to different anchors -- hand z is relative to that
hand's own wrist, pose z is relative to the hips), so plotting each
model's raw output makes the hand rigs look detached from the body rig.
To fix that, each hand's wrist joint is snapped onto the body rig's
matching wrist landmark (see anchor_hand_positions) -- the hand model only
contributes finger *shape*, the body model contributes *position*.

Controls:
  - Stand in front of the webcam; show one or two hands.
  - Press 'q' in the webcam window to quit.
"""

import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    PoseLandmarker, PoseLandmarkerOptions,
    PoseLandmarksConnections, RunningMode,
)
from vpython import canvas, sphere, cylinder, vector, rate, color

HAND_MODEL_PATH = "hand_landmarker.task"
POSE_MODEL_PATH = "pose_landmarker_lite.task"

NUM_HANDS = 2

# Bone connections between the 21 MediaPipe hand landmarks.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]

# Official 33-landmark body skeleton connections, straight from MediaPipe.
POSE_CONNECTIONS = [(c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS]

# MediaPipe's 33-point pose topology: indices of the two wrist landmarks.
POSE_WRIST_INDEX = {"Left": 15, "Right": 16}

# Physical size (in world units) the wider of the frame's two dimensions
# should map to, shared by both models so hands and body stay in scale
# with each other (see body_demo.py for why this beats a flat SCALE).
WORLD_SIZE = 20

HAND_JOINT_COLORS = [color.cyan, color.magenta]  # one per hand slot


def landmark_to_vector(landmark, frame_w, frame_h):
    unit = WORLD_SIZE / max(frame_w, frame_h)
    x = (landmark.x - 0.5) * frame_w * unit
    y = -(landmark.y - 0.5) * frame_h * unit
    z = -landmark.z * frame_w * unit
    return vector(x, y, z)


def make_rig(scene, num_joints, connections, joint_color, bone_color, joint_radius, bone_radius):
    joints = [sphere(canvas=scene, pos=vector(0, 0, 0), radius=joint_radius,
                      color=joint_color, visible=False)
              for _ in range(num_joints)]
    bones = [cylinder(canvas=scene, pos=vector(0, 0, 0), axis=vector(0, 0, 0),
                       radius=bone_radius, color=bone_color, visible=False)
             for _ in connections]
    return joints, bones


def set_rig_visible(joints, bones, visible):
    for joint in joints:
        joint.visible = visible
    for bone in bones:
        bone.visible = visible


def anchor_hand_positions(hand_positions, wrist_anchor):
    # hand_positions[0] is the hand model's own idea of where the wrist is.
    # Re-center the whole hand shape on that wrist, then shift it onto the
    # body model's wrist position instead, so the two rigs meet exactly.
    hand_wrist = hand_positions[0]
    return [wrist_anchor + (pos - hand_wrist) for pos in hand_positions]


def update_rig(joints, bones, positions, connections):
    for joint, pos in zip(joints, positions):
        joint.pos = pos
        joint.visible = True
    for bone, (i, j) in zip(bones, connections):
        bone.pos = positions[i]
        bone.axis = positions[j] - positions[i]
        bone.visible = True


def draw_landmarks_on_frame(frame, landmarks, connections, line_color, point_color):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for i, j in connections:
        cv2.line(frame, points[i], points[j], line_color, 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, point_color, -1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    pose_landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    ))
    hand_landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=NUM_HANDS,
    ))

    scene = canvas(title="Body + Hand Rig", width=800, height=600, background=color.black)

    pose_joints, pose_bones = make_rig(
        scene, num_joints=33, connections=POSE_CONNECTIONS,
        joint_color=color.orange, bone_color=color.white,
        joint_radius=0.3, bone_radius=0.12,
    )
    hand_rigs = [
        make_rig(
            scene, num_joints=21, connections=HAND_CONNECTIONS,
            joint_color=HAND_JOINT_COLORS[i % len(HAND_JOINT_COLORS)], bone_color=color.yellow,
            joint_radius=0.4, bone_radius=0.15,
        )
        for i in range(NUM_HANDS)
    ]

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

            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            wrist_anchors = {}
            if pose_result.pose_landmarks:
                landmarks = pose_result.pose_landmarks[0]
                positions = [landmark_to_vector(lm, frame_w, frame_h) for lm in landmarks]
                update_rig(pose_joints, pose_bones, positions, POSE_CONNECTIONS)
                draw_landmarks_on_frame(frame, landmarks, POSE_CONNECTIONS,
                                         line_color=(0, 255, 0), point_color=(0, 0, 255))
                wrist_anchors = {side: positions[idx] for side, idx in POSE_WRIST_INDEX.items()}
            else:
                set_rig_visible(pose_joints, pose_bones, False)

            detected_hands = hand_result.hand_landmarks
            detected_handedness = hand_result.handedness
            for i, (joints, bones) in enumerate(hand_rigs):
                if i < len(detected_hands):
                    landmarks = detected_hands[i]
                    positions = [landmark_to_vector(lm, frame_w, frame_h) for lm in landmarks]

                    side = detected_handedness[i][0].category_name if i < len(detected_handedness) else None
                    if side in wrist_anchors:
                        positions = anchor_hand_positions(positions, wrist_anchors[side])

                    update_rig(joints, bones, positions, HAND_CONNECTIONS)
                    draw_landmarks_on_frame(frame, landmarks, HAND_CONNECTIONS,
                                             line_color=(255, 0, 0), point_color=(0, 255, 255))
                else:
                    set_rig_visible(joints, bones, False)

            cv2.imshow("Webcam (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_landmarker.close()
        hand_landmarker.close()


if __name__ == "__main__":
    main()
