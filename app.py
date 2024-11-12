import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
from ultralytics import YOLO
from supervision import Detections
from Pipelines.inference_prediction import Predictions
import tensorflow as tf
import keras

# Custom CSS for gradient background
def set_background():
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(120deg, #E4EfE9 0%, #93A5CF 100%);
            background-attachment: fixed;
        }
        .stApp {
            background: linear-gradient(120deg, #E4EfE9, #93A5CF);
            color: #fff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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
            
    def detect_faces(self, image):
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

@keras.saving.register_keras_serializable()
def scaling(x, scale=1.0):
    return x * scale

# Load the model
model_inference = keras.models.load_model("artifacts/model.keras", custom_objects={'scaling': scaling})

class Predictions:
    def __init__(self, image_path, model=model_inference):
        self.image_path = image_path
        self.model = model
        self.input_size = (160, 160)
        
    def preprocess_image(self):
        image = Image.open(self.image_path).convert("RGB")
        image = image.resize(self.input_size, Image.LANCZOS)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    
    def predict(self):
        preprocessed_image = self.preprocess_image()
        prediction = self.model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction)
        return predicted_class

def present_names():
    present = []
    for files in os.listdir("artifacts/Inference"):
        image_path = f"artifacts/Inference/{files}"
        pred = Predictions(image_path)
        present.append(pred.predict())
    label = {0:"Benhur",1:"Rahul",2:"Santhosh",3:"Naveen"}
    mapped_names = [label[i] for i in present]
    return mapped_names

def main():
    # set_background()
    st.title("Automated Attendance")

    # Add camera input
    camera_input = st.camera_input("Take a picture")

    uploaded_file = st.file_uploader("Or choose an image...", type=["jpg", "jpeg", "png"])
    
    if camera_input is not None:
        # Use camera input image
        image = Image.open(camera_input)
    elif uploaded_file is not None:
        # Use uploaded image file
        image = Image.open(uploaded_file)
    else:
        st.write("Please upload an image or use the camera input.")
        return
    
    # Perform face detection and cropping
    cropper = ImageCropper()
    predicted_image = cropper.detect_faces(image)
    st.image(predicted_image, caption='Processed Image')
    
    # Save cropped images
    image_name = uploaded_file.name if uploaded_file else "camera_image"
    cropper.crop_and_save_images(image, image_name)
    
    # Predict present names
    names = present_names()
    st.write(f"Present names: {names}")

if __name__ == "__main__":
    main()
