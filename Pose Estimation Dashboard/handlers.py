import cv2
import numpy as np
import pandas as pd
import tempfile
import time
import streamlit as st
from PIL import Image
import mediapipe as mp

from pose_utils import landmarks_to_df, extract_all_angles
from plot_utils import plot_pose_3d, plot_pose_3d_animation

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Handle camera input (capture a single photo via Streamlit)
def handle_camera_input(model_complexity, min_detection_confidence, min_tracking_confidence):
    st.info("Please wait a few seconds until the camera is ready...")
    time.sleep(2)
    st.success("Camera is ready!")

    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        img = Image.open(camera_file)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        with mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        ) as pose:
            results = pose.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            process_results(results, img_cv)

# Handle file upload (image or video)
def handle_file_upload(model_complexity, min_detection_confidence, min_tracking_confidence):
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "png", "mp4", "mov", "avi"])
    if uploaded_file is not None:
        if uploaded_file.type.startswith("video"):            handle_video(uploaded_file, model_complexity, min_detection_confidence, min_tracking_confidence)
        else:
            img = Image.open(uploaded_file)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            with mp_pose.Pose(
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            ) as pose:
                results = pose.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                process_results(results, img_cv)

# Handle video uploads: process frames, collect landmarks and provide summary + animation
def handle_video(uploaded_file, model_complexity, min_detection_confidence, min_tracking_confidence):
    with st.spinner("Processing video..."):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_landmarks = []
        last_results, last_frame = None, None
        processed_frames = 0
        progress_bar = st.progress(0)

        with mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        ) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    all_landmarks.append(results.pose_landmarks.landmark)
                    last_results = results
                    last_frame = frame

                processed_frames += 1
                # protect division by zero
                if frame_count > 0:
                    progress_bar.progress(min(processed_frames / frame_count, 1.0))

        cap.release()
        tfile.close()

    st.success("Video processing completed!")

    col1, col2 = st.columns(2)
    with col1:
        st.video(uploaded_file, format="video/mp4")
    with col2:
        st.header("Analysis Results")
        if last_results and last_frame is not None:
            df = landmarks_to_df(last_results)
            angles = extract_all_angles(last_results, last_frame.shape)
            if df is not None:
                st.subheader("Landmarks Table")
                st.dataframe(df)

                st.subheader("Joint Angles")
                st.table(pd.DataFrame(angles, index=["Angle (degrees)"]).T)

                download_df = df.copy()
                download_df.loc[len(download_df)] = ["---", "---", "---", "---", "---"]
                for k, v in angles.items():
                    download_df.loc[len(download_df)] = [k, v, "-", "-", "-"]

                st.download_button(
                    "Download CSV Output",
                    download_df.to_csv(index=False).encode('utf-8'),
                    "pose_keypoints_angles.csv",
                    "text/csv",
                    help="Download detected landmarks and computed joint angles."
                )

    if all_landmarks:
        with st.expander("3D Body Animation", expanded=True):
            plot_pose_3d_animation(all_landmarks)

# Process and display results for image input
def process_results(results, img_cv):
    col1, col2 = st.columns(2)

    with col1:
        st.header("Original Image")
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Input Image")

    with col2:
        if results.pose_landmarks:
            st.header("Image with Landmarks")
            mp_drawing.draw_landmarks(img_cv, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Processed Result")


            df = landmarks_to_df(results)
            angles = extract_all_angles(results, img_cv.shape)

            if df is not None:
                st.subheader("Analysis Results")

                # Display joint angles
                st.markdown("##### Joint Angles")
                st.table(pd.DataFrame(angles, index=["Angle (degrees)"]).T)

                # Display landmarks
                st.markdown("##### Landmarks")
                st.dataframe(df)

                download_df = df.copy()
                download_df.loc[len(download_df)] = ["---", "---", "---", "---", "---"]
                for k, v in angles.items():
                    download_df.loc[len(download_df)] = [k, v, "-", "-", "-"]

                st.download_button(
                    "Download CSV Output",
                    download_df.to_csv(index=False).encode('utf-8'),
                    "pose_keypoints_angles.csv",
                    "text/csv",
                    help="Download detected landmarks and computed joint angles."
                )
        else:
            st.warning("No human pose detected. Please upload a clearer image.")
