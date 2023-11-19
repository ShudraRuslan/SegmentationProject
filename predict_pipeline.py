import os
import numpy as np
from PIL import Image
import tensorflow as tf

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gama=2, **kwargs):
        super(DiceLoss, self).__init__(**kwargs)
        self.name = 'NDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result


# Define the Intersection over Union (IoU) metric
def iou_metric(y_true, y_pred):
    intersection = tf.reduce_sum(tf.minimum(y_true, y_pred))
    union = tf.reduce_sum(tf.maximum(y_true, y_pred))
    iou = intersection / union
    return iou




# Load classification model
classification_model_path = 'classif_model.h5'
classification_model = tf.keras.models.load_model(classification_model_path)

# Load segmentation model
segmentation_model_path = 'segm_model.h5'
# Register the custom objects
custom_objects = {'DiceLoss': DiceLoss, 'iou_metric': iou_metric}

segmentation_model = tf.keras.models.load_model(segmentation_model_path, custom_objects=custom_objects)

def classify_and_segment_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)

            # Load and preprocess image for classification
            img = Image.open(image_path).resize((224, 224))
            img_array = np.array(img) / 255.0  # Normalize pixel values
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Perform classification
            classification_result = classification_model.predict(img_array)

            if classification_result[0][0] < 0.5:  # No ships detected
                result_message = "There are no ships in the image."
            else:
                # Reshape image for segmentation
                img_for_segmentation = img.resize((256, 256))
                img_array_segmentation = np.array(img_for_segmentation) / 255.0
                img_array_segmentation = np.expand_dims(img_array_segmentation, axis=0)

                # Perform segmentation
                segmented_image = segmentation_model.predict(img_array_segmentation)
                result_message = "Ship detected. Segmented image saved."

                # Save segmented image
                segmented_image_path = os.path.join(output_folder, f"segmented_{filename}")
                save_segmented_image(segmented_image[0], segmented_image_path)

            print(f"{filename}: {result_message}")

def save_segmented_image(segmented_image, output_path):
    # Process the segmented image as needed and save it
    segmented_image *= 255.0
    segmented_image = segmented_image.astype(np.uint8)

    # Squeeze to remove singleton dimensions
    segmented_image = np.squeeze(segmented_image)

    # Create a PIL Image for a single-channel (grayscale) image
    segmented_image = Image.fromarray(segmented_image, mode='L')

    # Save the image
    segmented_image.save(output_path)


# Example usage
input_folder = 'test_images'
output_folder = 'segment_results'
classify_and_segment_folder(input_folder, output_folder)
