import os
import numpy as np
from PIL import Image
import keras

class Predictions:
    @keras.saving.register_keras_serializable()
    def scaling(x, scale=1.0):
        return x * scale

    def __init__(self, model_path, image_path=None):
        self.image_path = image_path
        self.model_path = model_path
        self.model = keras.models.load_model(model_path, custom_objects={'scaling': self.scaling})
        self.input_size = (160, 160)

    def preprocess_image(self):
        try:
            image = Image.open(self.image_path).convert("RGB")
            image = image.resize(self.input_size, Image.LANCZOS)
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            return image_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def predict(self):
        preprocessed_image = self.preprocess_image()
        if preprocessed_image is not None:
            prediction = self.model.predict(preprocessed_image)
            predicted_class = np.argmax(prediction)
            return predicted_class
        else:
            return None

    def names(self, directory="R:/Attendance_system/artifacts/testing_images"):
        present = []
        for files in os.listdir(directory):
            image_path = os.path.join(directory, files)
            self.image_path = image_path  # Update image path for prediction
            predicted_class = self.predict()
            if predicted_class is not None:
                present.append(predicted_class)
        label = {0: "Ben", 1: "Rahul", 2: "Santhosh", 3: "Naveen"}
        mapped_names = [label[i] for i in present]
        return mapped_names

    def get_label(self, predicted_class):
        label = {0: "Benhur", 1: "Rahul", 2: "Santhosh", 3: "Naveen"}
        return label.get(predicted_class, "Unknown")

    def get_names(self, directory="R:/Attendance_system/artifacts/testing_images"):
        present = []
        for files in os.listdir(directory):
            image_path = os.path.join(directory, files)
            self.image_path = image_path  # Update image path for prediction
            predicted_class = self.predict()
            if predicted_class is not None:
                present.append(self.get_label(predicted_class))
        return present

# Example usage:
if __name__ == "__main__":
    predictions = Predictions("R:/Attendance_system/artifacts/model.keras")
    print(predictions.get_names())