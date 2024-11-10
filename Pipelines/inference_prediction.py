import numpy as np
from PIL import Image
import tensorflow as tf
import keras
import os

os.chdir("../")

@keras.saving.register_keras_serializable()
def scaling(x, scale=1.0):
    return x * scale

# Load the model
model = keras.models.load_model("artifacts/model.keras", custom_objects={'scaling': scaling})

import os
import numpy as np
from PIL import Image

class Predictions:
    def __init__(self, image_path = None, model = model):
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
    
    def names(self, directory):
        present = []
        
        for files in os.listdir(directory):
            image_path = os.path.join(directory, files)
            self.image_path = image_path  # Update image path for prediction
            present.append(self.predict())
            
        label = {0: "Ben", 1: "Rahul", 2: "Santhosh", 3: "Naveen"}
        mapped_names = [label[i] for i in present]
        return mapped_names

# Example usage:
predictions = Predictions()
predictions.names("artifacts/testing_images")