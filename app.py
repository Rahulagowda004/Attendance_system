import streamlit as st
from PIL import Image
import os
import io
import cv2
import numpy as np
from ultralytics import YOLO
from supervision import Detections

# Load the YOLO model
model = YOLO("models/yolo_model/model.pt")

class ImageCropper:
    def __init__(self, output_dir="artifacts/inference"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def crop_and_save_images(self, image, image_name):
        output = model(image)
        detections = Detections.from_ultralytics(output[0])
        for box_id, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, box)
            cropped_image = image.crop((x1, y1, x2, y2))

            cropped_image_filename = f"{os.path.splitext(image_name)[0]}_box_{box_id}.png"
            cropped_image_path = os.path.join(self.output_dir, cropped_image_filename)
            cropped_image.save(cropped_image_path)
            
    def detect_faces(self,image):
            output = model(image)
            face_boxes = []
            for result in output[0].boxes:
                if result.cls == 0:
                    x1, y1, x2, y2 = result.xyxy[0].tolist()
                    face_boxes.append((int(x1), int(y1), int(x2), int(y2)))
            image_np = np.array(image)
            for (x1, y1, x2, y2) in face_boxes:
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            final_image = Image.fromarray(image_np)
            return final_image

def main():
    st.title("YOLO Image Detection and Cropping")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        cropper = ImageCropper()
        predicted_image = cropper.detect_faces(image)
        st.image(predicted_image, caption='Uploaded Image', use_container_width=True) 
        cropper.crop_and_save_images(image, uploaded_file.name)

if __name__ == "__main__":
    main()