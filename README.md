# Pose Estimation Dashboard

![Pose Estimation Dashboard Banner](https://img.shields.io/badge/Project-Pose%20Estimation-blue?style=for-the-badge&logo=github)
![Python Version](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

## Project Description

The **Pose Estimation Dashboard** is a research and practical tool for analyzing human poses using **Computer Vision** techniques. This project utilizes the powerful **MediaPipe** library and the **Streamlit** framework to create an interactive dashboard that allows users to detect body poses from images and videos, visualize key landmarks, compute joint angles, and even view a 3D model of the body pose.

This tool can be used for various applications, including **sports motion analysis**, **body posture assessment systems**, and **academic research** in computer vision.

## Key Features

* **Diverse Input Support**: Process images, videos, or use a live webcam feed.
* **Accurate Pose Detection**: Detects and connects key body landmarks.
* **Joint Angle Calculation**: Computes key joint angles such as elbows and knees.
* **3D Visualization**: Displays a 3D body model using **Plotly**.
* **Data Export**: Allows downloading landmark and angle data in CSV format.
* **Interactive Dashboard**: Provides a simple and user-friendly interface with Streamlit.

## Technologies Used

* **Python**: The core programming language.
* **Streamlit**: For creating the web user interface.
* **MediaPipe**: For pose estimation.
* **OpenCV**: For image and video processing.
* **Plotly**: For displaying 3D pose models.
* **Pandas**: For data management and analysis.
* **NumPy**: For numerical computations.
* **Pillow**: For image handling.

## Installation and Setup

To run this project, follow these steps.

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/pose-estimation-dashboard.git](https://github.com/your-username/pose-estimation-dashboard.git)
    cd pose-estimation-dashboard
    ```

2.  **Create a virtual environment and install dependencies:**

    It is recommended to create a virtual environment to avoid conflicts with other packages.

    ```bash
    # Create the virtual environment (optional but recommended)
    python -m venv venv

    # Activate the virtual environment
    # On Windows
    venv\Scripts\activate
    # On macOS and Linux
    source venv/bin/activate

    # Install the required packages from requirements.txt
    pip install -r requirements.txt
    ```

3.  **Run the application:**

    After successful installation, run the application with the following command. This will open a local address (usually `http://localhost:8501`) in your browser, where you can interact with the dashboard.

    ```bash
    streamlit run app.py
    ```

## How to Use

1.  **Select Input Source:**
    * **Camera**: To take a photo using your webcam.
    * **Upload File**: To upload image (`.jpg`, `.png`) or video (`.mp4`, `.mov`, `.avi`) files.
2.  **Adjust Model Settings (Optional):**
    * In the sidebar, you can adjust MediaPipe model parameters like **Model Complexity**, **Detection Confidence**, and **Tracking Confidence** to fine-tune performance.
3.  **View Results:**
    * **Image**: After uploading an image or taking a photo, the processed image with landmarks, a table of landmark data, and joint angles will be displayed.
    * **Video**: After uploading a video, a 3D animation of body movements, along with data tables and angles, will be shown.
4.  **Download Data:**
    * The **"Download CSV Output"** button allows you to download the extracted landmark and joint angle data in a single CSV file.

## Contribution

This project is an excellent platform for learning and further development in computer vision. If you would like to contribute, please:

* Fork the repository.
* Create your changes in a new branch.
* Submit a **Pull Request** with a description of your changes.

## License

This project is released under the **MIT** license. For more information, please read the `LICENSE` file.
