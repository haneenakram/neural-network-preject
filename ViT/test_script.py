import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

from vit_classes import PatchCreation, PatchEmbedding, PositionEmbedding, TransformerEncoderBlock

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_trained_model(model_path):
    custom_objects = {
        'PatchCreation': PatchCreation,
        'PatchEmbedding': PatchEmbedding,
        'PositionEmbedding': PositionEmbedding,
        'TransformerEncoderBlock': TransformerEncoderBlock
    }
    return load_model(model_path, custom_objects=custom_objects)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Rescale to [0,1]
    return img_array

def load_images_from_list(image_paths):
    images = []
    valid_paths = []
    
    for img_path in image_paths:
        try:
            img_array = preprocess_image(img_path)
            images.append(img_array)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    if not images:
        raise ValueError("No valid images found in the provided list")
    
    return np.array(images), valid_paths

# def make_predictions(model, images, image_paths, class_indices_path):
#     """Make predictions on preprocessed images."""
#     with open(class_indices_path, 'r') as f:
#         class_indices = json.load(f)

#     idx_to_class = {v: k for k, v in class_indices.items()}
#     predictions = model.predict(images, batch_size=BATCH_SIZE, verbose=1)
#     predicted_classes = np.argmax(predictions, axis=1)
    
#     ids = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]
#     predicted_labels = [idx_to_class[pred] for pred in predicted_classes]

#     return pd.DataFrame({'id': ids, 'label': predicted_labels})
def make_predictions(model, images, image_paths, class_indices_path):
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)

    predictions = model.predict(images, batch_size=BATCH_SIZE, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    ids = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]
    
    # Use numeric class indices directly instead of mapping to class names
    predicted_labels = predicted_classes.tolist()

    return pd.DataFrame({'id': ids, 'label': predicted_labels})

def main(image_paths, model_path, output_file, class_indices_path):
    model = load_trained_model(model_path)
    images, valid_paths = load_images_from_list(image_paths)
    submission_df = make_predictions(model, images, valid_paths, class_indices_path)
    
    # Save results
    submission_df.to_csv(output_file, index=False)
    print(f"Predictions for {len(valid_paths)} images saved to {output_file}")

if __name__ == "__main__":
    MODEL_PATH = 'best_vit_model.keras'
    OUTPUT_FILE = 'exam_submission.csv'
    CLASS_INDICES_PATH = 'class_indices.json'
   
    import glob
    TEST_DIR = 'testNew'
    image_paths = glob.glob(os.path.join(TEST_DIR, '**', '*.jpg'), recursive=True)
    image_paths.extend(glob.glob(os.path.join(TEST_DIR, '**', '*.jpeg'), recursive=True))
    image_paths.extend(glob.glob(os.path.join(TEST_DIR, '**', '*.png'), recursive=True))
    
    if not image_paths:
        print(f"No image files found in {TEST_DIR}")
        exit(1)
    
    print(f"Found {len(image_paths)} images")
    main(image_paths, MODEL_PATH, OUTPUT_FILE, CLASS_INDICES_PATH)