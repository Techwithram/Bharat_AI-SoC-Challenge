import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.utils import class_weight
from keras.backend import epsilon
from keras.metrics import Recall
import numpy as np
import os

# --- 1. CONFIGURATION ---
IMG_SIZE = 128    # 128x128 is perfect for texture (Rot vs Ripe)
BATCH_SIZE = 32
EPOCHS_HEAD = 10  # Warm-up phase
EPOCHS_FINE = 20  # Increased slightly for better texture learning
DATASET_PATH = "C:\\Users\\krish\\Downloads\\Fruit_dataset" # UPDATE THIS IF NEEDED

# --- 2. CUSTOM FOCAL LOSS ---
# We use this instead of Manual Weights because it works better for 
# small defect detection (like black spots on a mango).
def focal_loss(gamma=2., alpha=4.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon_val = epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon_val, 1. - epsilon_val)
        y_true = tf.cast(y_true, tf.float32)
        
        alpha_t = y_true*alpha + (tf.ones_like(y_true)*(1-alpha))
        p_t = y_true*y_pred + (tf.ones_like(y_true)*(1-y_pred))

        fl = - alpha_t * tf.math.pow((tf.ones_like(y_true)-p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(fl)
    return focal_loss_fixed

# --- 3. DATA AUGMENTATION (FRUIT SPECIFIC) ---
# Fruits need aggressive rotation (no "up" or "down") and brightness (lighting changes).
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=180,      # Full rotation allowed
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,          
    horizontal_flip=True,
    vertical_flip=True,      # Tomato looks valid upside down
    brightness_range=[0.6, 1.4], # Simulates shadows/factory lighting
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

print("Loading Train Data...")
train_generator = train_datagen.flow_from_directory(
    f"{DATASET_PATH}/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print("Loading Validation Data...")
# Ensure your folder is named 'val' or 'test' and update below accordingly
val_generator = val_datagen.flow_from_directory(
    f"{DATASET_PATH}/test", # Changed to match your folder structure
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- 4. DYNAMIC CLASS DETECTION ---
# This implements the "Train 20" strategy.
num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())
print(f"✅ DETECTED {num_classes} CLASSES:")
print(class_names)

# Compute Class Weights (To balance Rare Rotten vs Common Ripe)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
weights_dict = dict(enumerate(class_weights))
print(f"Computed Weights: {weights_dict}")

# --- 5. BUILD MODEL (MobileNetV2) ---
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False # Freeze first

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x) 
predictions = Dense(num_classes, activation='softmax')(x) 

model = Model(inputs=base_model.input, outputs=predictions)

# --- 6. PHASE 1: TRAIN HEAD ---
print("\n--- PHASE 1: Training Output Layers ---")
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss=focal_loss(alpha=1.0), 
              metrics=['accuracy', Recall(name='recall')])

history_head = model.fit(
    train_generator,
    epochs=EPOCHS_HEAD,
    validation_data=val_generator,
    class_weight=weights_dict
)

# --- 7. PHASE 2: FINE-TUNING ---
print("\n--- PHASE 2: Fine-Tuning Textures ---")
base_model.trainable = True
# Unfreeze top 50 layers to learn "Rotten Texture" & "Ripeness Color"
for layer in base_model.layers[:-50]: 
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), # Very low LR for fine-tuning
              loss=focal_loss(alpha=1.0), 
              metrics=['accuracy', Recall(name='recall')])

history_fine = model.fit(
    train_generator,
    epochs=EPOCHS_FINE,
    validation_data=val_generator,
    class_weight=weights_dict
)

# --- 8. SAVE & CONVERT (FPGA OPTIMIZED) ---
print("\n--- Generating FPGA-Ready Model ---")
model.save("fruit_ripeness_model.h5")

# Representative Dataset for INT8 Quantization (Required for Zynq DPU)
def representative_data_gen():
    # Take a small batch of 100 images from the training set for calibration
    for _ in range(100):
        img_batch, _ = next(train_generator)
        yield [img_batch.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure Full Integer Quantization for FPGA DPU
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("fruit_model_quantized.tflite", "wb") as f:
    f.write(tflite_model)

print("SUCCESS: 'fruit_model_quantized.tflite' created.")
print(f"Dataset utilized: {num_classes} classes.")