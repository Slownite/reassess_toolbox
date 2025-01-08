import streamlit as st
import cv2
import os
import json
import subprocess
from PIL import Image, ImageDraw
import tempfile

# Set Streamlit page title
st.title("Video Child Labeling & Cropping Tool")

# Step 1: Upload video
uploaded_video = st.file_uploader(
    "Upload a video to label the first frame", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video:
    # Save the uploaded video in a temporary directory
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "uploaded_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.video(video_path)

    # Step 2: Extract first frame
    st.header("Step 2: Label the First Frame")

    # Extract first frame using OpenCV
    video_cap = cv2.VideoCapture(video_path)
    success, frame = video_cap.read()
    if success:
        frame_image_path = os.path.join(temp_dir, "first_frame.jpg")
        cv2.imwrite(frame_image_path, frame)
        frame_image = Image.open(frame_image_path)
        st.image(frame_image, caption="First Frame", use_column_width=True)
    else:
        st.error("Could not extract the first frame from the video.")
        st.stop()

    # Step 3: Label the child in the frame
    if 'bounding_box' not in st.session_state:
        st.session_state['bounding_box'] = {
            "x": 50, "y": 50, "width": 100, "height": 100}

    st.header("Step 3: Label the Child (Draw the Bounding Box)")

    # Use sliders to create the bounding box
    x = st.slider("X (Top-Left)", 0,
                  frame.shape[1], st.session_state['bounding_box']['x'])
    y = st.slider("Y (Top-Left)", 0,
                  frame.shape[0], st.session_state['bounding_box']['y'])
    width = st.slider(
        "Width", 1, frame.shape[1] - x, st.session_state['bounding_box']['width'])
    height = st.slider(
        "Height", 1, frame.shape[0] - y, st.session_state['bounding_box']['height'])

    # Update the session state
    st.session_state['bounding_box'] = {
        "x": x, "y": y, "width": width, "height": height}

    # Draw bounding box on the image
    frame_image_with_box = frame_image.copy()
    draw = ImageDraw.Draw(frame_image_with_box)
    draw.rectangle([x, y, x + width, y + height], outline="red", width=3)
    st.image(frame_image_with_box,
             caption="Labeled Frame with Bounding Box", use_column_width=True)

    # Export bounding box as JSON file
    if st.button("Export Bounding Box as JSON"):
        bounding_box_path = os.path.join(temp_dir, "bounding_box.json")
        with open(bounding_box_path, "w") as f:
            json.dump(st.session_state['bounding_box'], f)
        st.download_button("Download Bounding Box JSON", data=json.dumps(
            st.session_state['bounding_box']), file_name="bounding_box.json")

    # Step 4: Crop the video using FFmpeg
    st.header("Step 4: Crop the Video using the Bounding Box")

    if st.button("Crop Video"):
        output_video_path = os.path.join(temp_dir, "cropped_video.mp4")
        x, y, width, height = st.session_state['bounding_box']['x'], st.session_state['bounding_box'][
            'y'], st.session_state['bounding_box']['width'], st.session_state['bounding_box']['height']

        # FFmpeg command to crop video
        ffmpeg_command = f"ffmpeg -i {video_path} -vf 'crop={width}:{height}:{
            x}:{y}' -c:v libx264 -preset fast -crf 23 {output_video_path}"

        st.write("Running FFmpeg command:")
        st.code(ffmpeg_command, language='bash')

        try:
            # Run FFmpeg command
            subprocess.run(ffmpeg_command, shell=True, check=True)
            st.success(f"Video successfully cropped and saved as {
                       output_video_path}")

            # Download button for the cropped video
            with open(output_video_path, "rb") as f:
                st.download_button("Download Cropped Video",
                                   data=f, file_name="cropped_video.mp4")
        except subprocess.CalledProcessError as e:
            st.error(f"FFmpeg command failed: {e}")
