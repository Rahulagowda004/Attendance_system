import streamlit as st
import cv2
import os
import time
from PIL import Image
import numpy as np
from ultralytics import YOLO
from supervision import Detections
from training import training

class ImageCropper:
    def __init__(self, person_name):
        self.output_dir = os.path.join("artifacts/training", person_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = YOLO("models/yolo_model/model.pt")

    def process_image(self, image):
        """Process a single PIL Image object"""
        # Perform detection
        output = self.model(image)
        detections = Detections.from_ultralytics(output[0])

        # Store cropped images
        cropped_images = []
        
        # Iterate through each bounding box and save the cropped area
        for box_id, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers

            # Crop the image using the bounding box coordinates
            cropped_image = image.crop((x1, y1, x2, y2))
            
            # Generate unique filename using timestamp
            timestamp = image.filename if hasattr(image, 'filename') else str(box_id)
            cropped_image_filename = f"crop_{timestamp}_box_{box_id}.png"
            cropped_image_path = os.path.join(self.output_dir, cropped_image_filename)
            
            # Save the cropped image
            cropped_image.save(cropped_image_path)
            cropped_images.append(cropped_image_path)
            
        return cropped_images

def main():
    st.title("Image Capture and Processing App")
    
    # Initialize session state variables
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'capturing' not in st.session_state:
        st.session_state.capturing = False
    if 'person_name' not in st.session_state:
        st.session_state.person_name = ""
    
    # Get person's name if not already captured
    if not st.session_state.person_name:
        with st.form("name_form"):
            name_input = st.text_input("Please enter your name:")
            submit_button = st.form_submit_button("Submit")
            if submit_button and name_input:
                st.session_state.person_name = name_input.strip()
    
    # Only show the rest of the interface if we have the person's name
    if st.session_state.person_name:
        # Initialize ImageCropper with person's name
        cropper = ImageCropper(st.session_state.person_name)
        
        # Create a placeholder for the camera feed
        camera_placeholder = st.empty()
        
        # Create capture button
        if st.button("Capture 100 Images"):
            st.session_state.capturing = True
            st.session_state.frame_count = 0
        
        # Create a placeholder for the progress bar
        progress_placeholder = st.empty()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera")
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the camera feed
            camera_placeholder.image(rgb_frame, channels="RGB")
            
            # Capture and process images if capturing is active
            if st.session_state.capturing and st.session_state.frame_count < 100:
                # Convert frame to PIL Image
                pil_image = Image.fromarray(rgb_frame)
                
                # Process and save the image
                timestamp = int(time.time() * 1000)
                pil_image.filename = f"frame_{timestamp}.png"
                
                # Process image using the cropper
                cropped_images = cropper.process_image(pil_image)
                
                st.session_state.frame_count += 1
                
                # Update progress bar
                progress = st.session_state.frame_count / 100
                progress_placeholder.progress(progress)
                
                if st.session_state.frame_count >= 100:
                    st.session_state.capturing = False
                    progress_placeholder.empty()
                    st.success(f"Finished capturing images for {st.session_state.person_name}!")
                    break
                    
            # Add a small delay to prevent overwhelming the system
            time.sleep(0.1)
        
        # Release the camera when done
        cap.release()
        
        training()
if __name__ == "__main__":
    main()