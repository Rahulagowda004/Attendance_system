from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image, ImageDraw
import os


class inference_cropping:
    def __init__(self, image_path = " provide the path", output_directory  = "artifacts/inference"):
        self.model = YOLO("models\\yolo_model\\model.pt")
        self.image_path = image_path
        self.output_dir = output_directory
        
        image = Image.open(self.image_path)
        output = self.model(image)
        detections = Detections.from_ultralytics(output[0])

        # Create directory to save cropped images
        os.makedirs(self.output_dir, exist_ok=True)

        # Iterate through each bounding box and save the cropped area
        for box_id, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers

            # Crop the image using the bounding box coordinates
            cropped_image = image.crop((x1, y1, x2, y2))
            
            # Save the cropped image
            cropped_image_path = os.path.join(self.output_dir, f"box_{box_id}.png")
            cropped_image.save(cropped_image_path)

        print(f"All bounding boxes cropped and saved in '{self.output_dir}' directory.")