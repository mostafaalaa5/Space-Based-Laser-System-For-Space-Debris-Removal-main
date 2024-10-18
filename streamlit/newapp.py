import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import tempfile
import os

# Load YOLOv8 model
model = YOLO('best.pt')

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.capture_image = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img, _ = detect_objects(img)
        if self.capture_image:
            self.captured_image = img.copy()
            self.capture_image = False
        return img

    def get_captured_image(self):
        return getattr(self, 'captured_image', None)

def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def detect_objects(image):
    results = model(image)
    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (95, 207, 30), 3)
            label = f"{result.names[int(bbox.cls[0])]}: {bbox.conf[0]:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return image, results

def run():
    # Sidebar
    st.sidebar.image('str.png', use_column_width=True)
    st.sidebar.title("Satellite Detection")
    st.sidebar.markdown("### Choose Input Source")
    activities = ["Image", "Webcam", "Video"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)

    # Main section
    st.markdown("<h1 style='text-align: center; color: #FF5733;'>Satellite Detection using YOLOv8</h1>", unsafe_allow_html=True)
    
    if choice == 'Image':
        st.markdown("### Upload Images for Detection")
        img_files = st.file_uploader("Choose Images", type=['jpg', 'jpeg', 'jfif', 'png'], accept_multiple_files=True)
        
        if img_files:
            for img_file in img_files:
                img = np.array(Image.open(img_file))
                original_img = img.copy()  # Create a copy of the original image
                processed_img, results = detect_objects(img)

                # Display images side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_img, caption='Original Image', use_column_width=True)
                with col2:
                    st.image(processed_img, caption='Processed Image', use_column_width=True)
                
                # Image comparison with zoom option
                with st.expander("Zoom Processed Image"):
                    st.image(processed_img, caption='Zoomed Processed Image', use_column_width=True)
                
                result_image = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                st.markdown(get_image_download_link(result_image, img_file.name, 'Download Image'), unsafe_allow_html=True)
                for result in results:
                    for bbox in result.boxes:
                        st.markdown(f"<p style='color: #4CAF50;'>Class: {result.names[int(bbox.cls[0])]}, Confidence: {bbox.conf[0]:.2f}</p>", unsafe_allow_html=True)

    elif choice == 'Webcam':
        st.markdown("### Real-time Object Detection with Webcam")
        ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

        if st.button("Capture Image"):
            ctx.video_transformer.capture_image = True

        if ctx.video_transformer and ctx.video_transformer.get_captured_image() is not None:
            captured_image = ctx.video_transformer.get_captured_image()
            original_img = captured_image.copy()  # Create a copy of the captured image
            processed_img, results = detect_objects(captured_image)

            # Display images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_img, caption='Captured Image', use_column_width=True)
            with col2:
                st.image(processed_img, caption='Processed Image', use_column_width=True)
            
            # Image comparison with zoom option
            with st.expander("Zoom Processed Image"):
                st.image(processed_img, caption='Zoomed Processed Image', use_column_width=True)

            result_image = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            st.markdown(get_image_download_link(result_image, 'captured_image.jpg', 'Download Image'), unsafe_allow_html=True)
            for result in results:
                for bbox in result.boxes:
                    st.markdown(f"<p style='color: #4CAF50;'>Class: {result.names[int(bbox.cls[0])]}, Confidence: {bbox.conf[0]:.2f}</p>", unsafe_allow_html=True)

    elif choice == 'Video':
        st.markdown("### Upload a Video for Detection")
        video_file = st.file_uploader("Choose a Video", type=['mp4', 'mov', 'avi', 'mkv'])
        
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame, results = detect_objects(frame)
                stframe.image(frame, channels='BGR')
            cap.release()
            os.remove(tfile.name)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("This application uses YOLOv8 for object detection in images, webcam, and video files. Upload your files or use the webcam for real-time detection.")

run()
