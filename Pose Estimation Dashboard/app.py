import streamlit as st
import nbformat
import io
import base64

from handlers import handle_camera_input, handle_file_upload

# Streamlit page configuration
st.set_page_config(
    page_title="Pose Estimation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Pose Estimation Dashboard")

    # Introduction section
    with st.expander("Introduction and Guide", expanded=True):
        st.markdown("""
        ### Welcome to the Pose Estimation Dashboard!
        This is a research-driven project for analyzing human body poses.
        With this tool you can:
        - Detect body pose from images or videos
        - Visualize body landmarks
        - Compute joint angles such as elbows and knees
        - Export data as CSV
        - Display 3D human pose models
        """)

    st.markdown("---")

    # Scientific Background Section
    with st.expander("Scientific Background and References", expanded=False):
        st.markdown("""
        ### 1. Overview of Human Pose Estimation (HPE)
        Human Pose Estimation (HPE) is a vital area in **Computer Vision** that identifies and tracks body keypoints (landmarks).
        It is divided into two main categories:
        - **2D HPE**: Detects body joints on the image plane (x, y).
        - **3D HPE**: Estimates joints in 3D space (x, y, z). This is more accurate for motion analysis.
        
        Tools like **OpenPose**, **DeepPose**, and **MediaPipe Pose** are widely used in this field.
        
        **MediaPipe Pose**, used in this project, supports both 2D and 3D pose estimation from a single camera.
        However, z-axis estimation is less accurate and more sensitive to noise.
        """)
        
        st.markdown("---")
        st.markdown("""
        ### 2. Advanced Methods to Improve Accuracy
        To address challenges of single-camera HPE, researchers propose:
        - **Multi-camera data fusion**: Using multiple cameras and triangulation.
        - **Filtering methods**: Kalman filter and smoothing to reduce noise and stabilize tracking.
        """)
        
        st.markdown("---")
        st.markdown("""
        ### 3. Project Architecture and Related Work
        - **Angle Calculation**: Computes elbow and knee angles from landmarks.
        - **3D Visualization with Plotly**: Displays 3D body models interactively.
        
        This project demonstrates practical application of research ideas for **exercise assessment systems**.
        """)
        
        st.markdown("---")
        st.markdown("""
        ### 4. Upload and Display Jupyter Notebook
        You can upload a Jupyter Notebook (`.ipynb`) and preview its content below.
        """)
        
        uploaded_notebook = st.file_uploader("Upload a Jupyter Notebook", type=["ipynb"])
        if uploaded_notebook is not None:
            st.subheader("Notebook Content")
            try:
                notebook = nbformat.read(uploaded_notebook, as_version=4)
                
                with st.expander("Show Notebook Content", expanded=True):
                    for cell in notebook.cells:
                        if cell.cell_type == 'markdown':
                            st.markdown("".join(cell.source))
                        elif cell.cell_type == 'code':
                            st.code("".join(cell.source), language='python')
                            if cell.outputs:
                                for output in cell.outputs:
                                    if output.output_type == 'stream':
                                        st.write(output.text)
                                    elif output.output_type == 'display_data':
                                        if 'text/plain' in output.data:
                                            st.write(output.data['text/plain'])
                                        if 'image/png' in output.data:
                                            st.image(io.BytesIO(base64.b64decode(output.data['image/png'])))
                                    elif output.output_type == 'execute_result':
                                        if 'text/plain' in output.data:
                                            st.write(output.data['text/plain'])
                                    elif output.output_type == 'error':
                                        st.error("\\n".join(output.traceback))
                        st.markdown("---")
            except Exception as e:
                st.error(f"Error while reading notebook: {e}")

    st.markdown("---")

    # Sidebar settings
    st.sidebar.header("Model Settings")
    model_complexity = st.sidebar.selectbox("Model Complexity", [0, 1, 2], index=1)
    min_detection_confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5)
    min_tracking_confidence = st.sidebar.slider("Tracking Confidence", 0.0, 1.0, 0.5)

    # Input mode selection
    mode = st.radio("Choose Input Source:", ["Camera", "Upload File"], help="Select an input method.")

    if mode == "Camera":
        handle_camera_input(model_complexity, min_detection_confidence, min_tracking_confidence)
    elif mode == "Upload File":
        handle_file_upload(model_complexity, min_detection_confidence, min_tracking_confidence)

if __name__ == "__main__":
    main()
