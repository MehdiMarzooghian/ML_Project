import numpy as np
import pandas as pd
import mediapipe as mp

mp_pose = mp.solutions.pose

# Convert MediaPipe landmarks to a pandas DataFrame
def landmarks_to_df(results):
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        data = []
        for i, lm in enumerate(landmarks):
            data.append([
                mp_pose.PoseLandmark(i).name,
                lm.x, lm.y, lm.z, lm.visibility
            ])
        df = pd.DataFrame(data, columns=["Landmark", "X", "Y", "Z", "Visibility"])
        return df
    return None

# Calculate the angle between three points in 2D space (in degrees)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Extract common joint angles (elbows and knees) from MediaPipe results
def extract_all_angles(results, frame_shape):
    h, w, _ = frame_shape
    angles = {}
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Left Elbow
        angles["Left Elbow Angle"] = calculate_angle(
            [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h],
            [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h],
            [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
        )

        # Right Elbow
        angles["Right Elbow Angle"] = calculate_angle(
            [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h],
            [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h],
            [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        )

        # Left Knee
        angles["Left Knee Angle"] = calculate_angle(
            [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h],
            [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h],
            [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
        )

        # Right Knee
        angles["Right Knee Angle"] = calculate_angle(
            [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h],
            [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h],
            [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h]
        )

    return angles
