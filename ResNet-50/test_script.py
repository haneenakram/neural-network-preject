import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models, regularizers

# ---------------------- Configuration ----------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 7

# ---------------------- Model Definition ----------------------
def build_resnet50_custom(input_shape=(224, 224, 3), num_classes=7):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', name='conv1_conv',
                      kernel_regularizer=regularizers.l2(0.0005))(inputs)
    x = layers.BatchNormalization(axis=3, name='conv1_bn')(x)
    x = layers.ReLU(name='conv1_relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same', name='pool1_pool')(x)

    def bottleneck_block(x, filters, stage, block, downsample=False):
        filters1, filters2, filters3 = filters, filters, filters * 4
        conv_base = f'conv{stage}_block{block}'

        strides = 2 if downsample else 1
        shortcut = x

        x = layers.Conv2D(filters1, (1, 1), strides=strides, name=f'{conv_base}_1_conv',
                          kernel_regularizer=regularizers.l2(0.0005))(x)
        x = layers.BatchNormalization(axis=3, name=f'{conv_base}_1_bn')(x)
        x = layers.ReLU(name=f'{conv_base}_1_relu')(x)

        x = layers.Conv2D(filters2, (3, 3), padding='same', name=f'{conv_base}_2_conv',
                          kernel_regularizer=regularizers.l2(0.0005))(x)
        x = layers.BatchNormalization(axis=3, name=f'{conv_base}_2_bn')(x)
        x = layers.ReLU(name=f'{conv_base}_2_relu')(x)

        x = layers.Conv2D(filters3, (1, 1), name=f'{conv_base}_3_conv',
                          kernel_regularizer=regularizers.l2(0.0005))(x)
        x = layers.BatchNormalization(axis=3, name=f'{conv_base}_3_bn')(x)

        if downsample or shortcut.shape[-1] != filters3:
            shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                                     name=f'{conv_base}_0_conv',
                                     kernel_regularizer=regularizers.l2(0.0005))(shortcut)
            shortcut = layers.BatchNormalization(axis=3, name=f'{conv_base}_0_bn')(shortcut)

        x = layers.add([x, shortcut], name=f'{conv_base}_add')
        x = layers.ReLU(name=f'{conv_base}_out')(x)
        return x

    block_config = [3, 4, 6, 3]
    filters = [64, 128, 256, 512]
    block_id = 1

    for stage, reps in enumerate(block_config, start=2):
        for b in range(reps):
            x = bottleneck_block(x, filters[stage - 2], stage, block_id, downsample=(b == 0))
            block_id += 1

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation='relu', name='fc1',
                     kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs, outputs, name='CustomResNet50')
    return model

# ---------------------- Preprocessing ----------------------
def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
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

# ---------------------- Inference ----------------------
def make_predictions(model, images, image_paths, class_indices_path):
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)

    predictions = model.predict(images, batch_size=BATCH_SIZE, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    ids = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]
    predicted_labels = predicted_classes.tolist()

    return pd.DataFrame({'id': ids, 'label': predicted_labels})

# ---------------------- Main Entry ----------------------
def main(image_paths, weights_path, output_file, class_indices_path):
    model = build_resnet50_custom(input_shape=(224, 224, 3), num_classes=NUM_CLASSES)
    model.load_weights(weights_path)
    print("✅ Model weights loaded.")
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

    images, valid_paths = load_images_from_list(image_paths)
    submission_df = make_predictions(model, images, valid_paths, class_indices_path)

    submission_df.to_csv(output_file, index=False)
    print(f"✅ Predictions for {len(valid_paths)} images saved to {output_file}")

# ---------------------- Run Script ----------------------
if __name__ == "__main__":
    WEIGHTS_PATH = 'resnet.weights.h5'          # Trained weights
    OUTPUT_FILE = 'submission.csv'           # Output CSV
    CLASS_INDICES_PATH =  '../class_indices.json'      # Class label mapping
    TEST_DIR = '../testNew'                             # Test images directory

    import glob
    image_paths = glob.glob(os.path.join(TEST_DIR, '**', '*.jpg'), recursive=True)
    image_paths.extend(glob.glob(os.path.join(TEST_DIR, '**', '*.jpeg'), recursive=True))
    image_paths.extend(glob.glob(os.path.join(TEST_DIR, '**', '*.png'), recursive=True))

    if not image_paths:
        print(f"No image files found in {TEST_DIR}")
        exit(1)

    print(f"🔍 Found {len(image_paths)} images")
    main(image_paths, WEIGHTS_PATH, OUTPUT_FILE, CLASS_INDICES_PATH)
