import os
from ultralytics import YOLO
from supervision import Detections
from PIL import Image

# Download and load model
model = YOLO("models/yolo_model/model.pt")

class ImageCropper:
    def __init__(self, output_dir="artifacts/training"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def crop_images(self, image_paths):
        # Iterate through each image in the provided list
        for image_path in image_paths:
            # Check if the file is an image
            if image_path.endswith((".png", ".jpg", ".jpeg")):
                image = Image.open(image_path)

                # Load model and perform detection
                output = model(image)
                detections = Detections.from_ultralytics(output[0])

                # Iterate through each bounding box and save the cropped area
                for box_id, box in enumerate(detections.xyxy):
                    x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers

                    # Crop the image using the bounding box coordinates
                    cropped_image = image.crop((x1, y1, x2, y2))

                    # Save the cropped image with a unique filename
                    cropped_image_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_box_{box_id}.png"
                    cropped_image_path = os.path.join(self.output_dir, cropped_image_filename)
                    cropped_image.save(cropped_image_path)

# Example usage
if __name__ == "__main__":
    image_paths = ["pic.png"]  # Replace with your image paths
    output_directory = "artifacts/training"  # Replace with your output folder path
    cropper = ImageCropper(output_directory)
    cropper.crop_images(image_paths)