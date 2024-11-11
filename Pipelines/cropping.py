import os
from ultralytics import YOLO
from supervision import Detections
from PIL import Image, ImageDraw

# download and load model
model = YOLO("models\\yolo_model\\model.pt")

class ImageCropper:
    def __init__(self, input_dir = "put ur path", output_dir = "artifacts\training"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def crop_images(self):
        # Iterate through each image in the input directory
        for image_filename in os.listdir(self.input_dir):
            # Check if the file is an image
            if image_filename.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(self.input_dir, image_filename)
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
                    cropped_image_filename = f"{os.path.splitext(image_filename)[0]}_box_{box_id}.png"
                    cropped_image_path = os.path.join(self.output_dir, cropped_image_filename)
                    cropped_image.save(cropped_image_path)

# Example usage
if __name__ == "__main__":
    input_directory = "testing_images"  # Replace with your input folder path
    output_directory = "artifacts/testing_images"  # Replace with your output folder path
    cropper = ImageCropper(input_directory, output_directory)
    cropper.crop_images()