"""
Same body + hand tracking demo as body_hand_demo.py, but pointed at an
iPhone used as a webcam instead of a built-in/USB webcam.

IMPORTANT: plugging an iPhone into Windows over USB is NOT enough by
itself -- Windows only sees it as a file-transfer device, not a camera.
You need a bridge app running that exposes the phone as a virtual webcam
(e.g. Camo, iVCam, EpocCam). Once that's running, the phone shows up to
Windows as an ordinary camera device, just at a different index than
whatever OpenCV's default index 0 normally opens.

There's no reliable way to know that index in advance -- it depends on
which bridge app you used and what else is plugged in. Set CAMERA_INDEX
below and try running this; if it can't open that index (or opens the
wrong camera), it will list every camera index it can actually read a
frame from so you can pick the right one.

Controls:
  - Stand in front of the iPhone; show one or two hands.
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

# Index of the camera device to open. 0 is usually a laptop's built-in
# webcam -- once your iPhone bridge app (Camo/iVCam/EpocCam/etc.) is
# running, the phone will likely show up as 1, 2, or higher. If this is
# wrong, this script will list working indices for you on startup.
CAMERA_INDEX = 1
CAMERA_PROBE_RANGE = range(5)  # indices to check when CAMERA_INDEX fails

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

# MediaPipe's 33-point pose topology: indices of the wrist/elbow landmarks.
POSE_WRIST_INDEX = {"Left": 15, "Right": 16}
POSE_ELBOW_INDEX = {"Left": 13, "Right": 14}

# MediaPipe's 21-point hand topology: wrist and middle-finger-base indices,
# used as a stable "palm length" reference for rescaling (see hand_to_forearm_scale).
HAND_WRIST_INDEX = 0
HAND_MIDDLE_MCP_INDEX = 9

# Rough real-world ratio of palm length (wrist to middle-finger base) to
# forearm length (elbow to wrist), used to size hands relative to the body
# instead of relative to how close the hand happens to be to the camera.
# Tune this by eye if hands still look too big/small.
HAND_TO_FOREARM_RATIO = 0.1

# Joint/bone radii for hands are drawn as a fraction of that hand's own
# (rescaled) palm length rather than a fixed size -- otherwise a fixed
# radius that looked fine at the hand model's raw, oversized scale ends up
# bigger than the whole rescaled hand, and every joint blurs into one blob.
HAND_JOINT_RADIUS_FRACTION = 0.12
HAND_BONE_RADIUS_FRACTION = 0.045
MIN_HAND_RADIUS = 0.02  # floor so a momentary bad reading can't zero it out

# Running two model inferences per frame is expensive. Body movement is
# coarse and slow compared to finger articulation, so the pose model can
# run at a fraction of the rate without looking wrong -- freeing up time
# for the hand model, which actually needs to keep up with fast motion.
POSE_EVERY_N_FRAMES = 2

# Physical size (in world units) the wider of the frame's two dimensions
# should map to, shared by both models so hands and body stay in scale
# with each other (see body_demo.py for why this beats a flat SCALE).
WORLD_SIZE = 20

HAND_JOINT_COLORS = [color.cyan, color.magenta]  # one per hand slot


def find_working_camera_indices(max_index):
    working = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                working.append((i, w, h))
        cap.release()
    return working


def open_camera(preferred_index, probe_range):
    cap = cv2.VideoCapture(preferred_index)
    if cap.isOpened():
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()

    print(f"Could not read frames from camera index {preferred_index}.")
    print("Checking other camera indices...")
    working = find_working_camera_indices(max(probe_range) + 1 if probe_range else 5)
    if not working:
        print("No camera indices produced a readable frame. Is your iPhone's")
        print("bridge app (Camo/iVCam/EpocCam/etc.) actually running?")
    else:
        print("Found working camera(s):")
        for idx, w, h in working:
            print(f"  index {idx}: {w}x{h}")
        print("Update CAMERA_INDEX at the top of this file to the one that's your iPhone.")
    return None


def landmark_to_vector(landmark, frame_w, frame_h):
    unit = WORLD_SIZE / max(frame_w, frame_h)
    x = (landmark.x - 0.5) * frame_w * unit
    y = -(landmark.y - 0.5) * frame_h * unit
    z = -landmark.z * frame_w * unit
    return vector(x, y, z)


def distance(a, b):
    return (a - b).mag


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


def set_rig_radius(joints, bones, joint_radius, bone_radius):
    for joint in joints:
        joint.radius = joint_radius
    for bone in bones:
        bone.radius = bone_radius


def hand_to_forearm_scale(hand_positions, forearm_length):
    # The hand model's apparent size only reflects how close that hand is
    # to the camera, not its real size relative to the body. Rescale it so
    # its palm length matches a plausible fraction of this arm's own
    # (body-model-measured) forearm length instead.
    palm_length = distance(hand_positions[HAND_WRIST_INDEX], hand_positions[HAND_MIDDLE_MCP_INDEX])
    if palm_length < 1e-6:
        return 1.0
    return (forearm_length * HAND_TO_FOREARM_RATIO) / palm_length


def anchor_hand_positions(hand_positions, wrist_anchor, scale=1.0):
    # hand_positions[0] is the hand model's own idea of where the wrist is.
    # Re-center the whole hand shape on that wrist, rescale it, then shift
    # it onto the body model's wrist position instead, so the two rigs
    # meet exactly and at a consistent, body-relative size.
    hand_wrist = hand_positions[0]
    return [wrist_anchor + (pos - hand_wrist) * scale for pos in hand_positions]


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
    cap = open_camera(CAMERA_INDEX, CAMERA_PROBE_RANGE)
    if cap is None:
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

    scene = canvas(title="Body + Hand Rig (iPhone camera)", width=800, height=600, background=color.black)

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
    prev_frame_time = start_time
    fps_ema = None
    frame_index = 0
    wrist_anchors = {}
    forearm_lengths = {}
    last_pose_landmarks = None

    try:
        while True:
            rate(30)  # let VPython process/render this frame

            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)

            if frame_index % POSE_EVERY_N_FRAMES == 0:
                pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                if pose_result.pose_landmarks:
                    landmarks = pose_result.pose_landmarks[0]
                    positions = [landmark_to_vector(lm, frame_w, frame_h) for lm in landmarks]
                    update_rig(pose_joints, pose_bones, positions, POSE_CONNECTIONS)
                    wrist_anchors = {side: positions[idx] for side, idx in POSE_WRIST_INDEX.items()}
                    forearm_lengths = {
                        side: distance(positions[POSE_ELBOW_INDEX[side]], positions[idx])
                        for side, idx in POSE_WRIST_INDEX.items()
                    }
                    last_pose_landmarks = landmarks
                else:
                    set_rig_visible(pose_joints, pose_bones, False)
                    wrist_anchors = {}
                    forearm_lengths = {}
                    last_pose_landmarks = None

            if last_pose_landmarks is not None:
                draw_landmarks_on_frame(frame, last_pose_landmarks, POSE_CONNECTIONS,
                                         line_color=(0, 255, 0), point_color=(0, 0, 255))

            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            detected_hands = hand_result.hand_landmarks
            detected_handedness = hand_result.handedness
            for i, (joints, bones) in enumerate(hand_rigs):
                if i < len(detected_hands):
                    landmarks = detected_hands[i]
                    positions = [landmark_to_vector(lm, frame_w, frame_h) for lm in landmarks]

                    side = detected_handedness[i][0].category_name if i < len(detected_handedness) else None
                    if side in wrist_anchors:
                        scale = hand_to_forearm_scale(positions, forearm_lengths[side])
                        positions = anchor_hand_positions(positions, wrist_anchors[side], scale)

                    effective_palm_length = distance(positions[HAND_WRIST_INDEX], positions[HAND_MIDDLE_MCP_INDEX])
                    set_rig_radius(
                        joints, bones,
                        joint_radius=max(effective_palm_length * HAND_JOINT_RADIUS_FRACTION, MIN_HAND_RADIUS),
                        bone_radius=max(effective_palm_length * HAND_BONE_RADIUS_FRACTION, MIN_HAND_RADIUS),
                    )
                    update_rig(joints, bones, positions, HAND_CONNECTIONS)
                    draw_landmarks_on_frame(frame, landmarks, HAND_CONNECTIONS,
                                             line_color=(255, 0, 0), point_color=(0, 255, 255))
                else:
                    set_rig_visible(joints, bones, False)

            now = time.time()
            instant_fps = 1.0 / (now - prev_frame_time) if now > prev_frame_time else 0.0
            prev_frame_time = now
            fps_ema = instant_fps if fps_ema is None else (0.9 * fps_ema + 0.1 * instant_fps)
            cv2.putText(frame, f"FPS: {fps_ema:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("iPhone camera (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_landmarker.close()
        hand_landmarker.close()


if __name__ == "__main__":
    main()
