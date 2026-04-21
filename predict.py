import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model = tf.keras.models.load_model("rice_disease_model.h5")

# Class names (must match your folder names in alphabetical order)
class_names = ["bacterial_leaf_blight", "brownspot", "leaf_smut"]


def predict_disease(image_path):
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print(f"\n🧪 Image: {os.path.basename(image_path)}")
    print(f"🔍 Predicted Disease: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2f}%")


# Example usage
if __name__ == "__main__":
    # Put your test image path here
    test_image = "test_image.jpg"  # ← CHANGE THIS
    predict_disease(test_image)
