import streamlit as st
from PIL import Image
import os
import io
from ultralytics import YOLO
from supervision import Detections

# Load the YOLO model
model = YOLO("models/yolo_model/model.pt")

class ImageCropper:
    def __init__(self, output_dir="artifacts/inference"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def crop_and_save_images(self, image, image_name):
        # Run YOLO detection on the image
        output = model(image)
        detections = Detections.from_ultralytics(output[0])

        # Save each cropped bounding box region
        for box_id, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers

            # Crop the image using the bounding box coordinates
            cropped_image = image.crop((x1, y1, x2, y2))

            # Save the cropped image with a unique filename
            cropped_image_filename = f"{os.path.splitext(image_name)[0]}_box_{box_id}.png"
            cropped_image_path = os.path.join(self.output_dir, cropped_image_filename)
            cropped_image.save(cropped_image_path)

def main():
    st.title("YOLO Image Detection and Cropping")

    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)  # Updated here
        
        # Display image details
        st.write("Image Info:")
        st.write(f"Format: {image.format}")
        st.write(f"Size: {image.size}")
        st.write(f"Mode: {image.mode}")
        
        # Initialize ImageCropper and save cropped images
        cropper = ImageCropper()
        cropper.crop_and_save_images(image, uploaded_file.name)

if __name__ == "__main__":
    main()
