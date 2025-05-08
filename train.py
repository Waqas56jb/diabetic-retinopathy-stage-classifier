import tensorflow as tf
from tensorflow.keras.applications import convnext
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import random
import shutil

# Define paths to dataset folders (relative to train.py)
train_dir = 'train'  # Directly in the same directory as train.py
val_dir = 'val'      # Directly in the same directory as train.py
test_dir = 'test'    # Directly in the same directory as train.py

# Create a temporary directory for a subset of 50 images (10 from each class)
temp_train_dir = 'temp_train'
if os.path.exists(temp_train_dir):
    shutil.rmtree(temp_train_dir)
os.makedirs(temp_train_dir)

# Copy 10 images from each class to the temporary directory
class_names = {0: 'No_Dr', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative_DR'}
num_classes = len(class_names)
images_per_class = 10

for class_id in range(num_classes):
    class_dir = os.path.join(train_dir, str(class_id))
    temp_class_dir = os.path.join(temp_train_dir, str(class_id))
    os.makedirs(temp_class_dir)
    all_images = os.listdir(class_dir)
    selected_images = random.sample(all_images, min(images_per_class, len(all_images)))
    for img in selected_images[:10]:  # Limit to 10 images per class
        shutil.copy(os.path.join(class_dir, img), os.path.join(temp_class_dir, img))

# Image dimensions and batch size
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 8  # Reduced batch size for efficiency

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    zoom_range=0.2
)

# Data generators for validation and test (no augmentation, just rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets using ImageDataGenerator
train_generator = train_datagen.flow_from_directory(
    temp_train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Function to build ConvNeXtTiny model
def build_convnext_tiny_model():
    base_model = convnext.ConvNeXtTiny(
        weights='imagenet',  # Using pre-trained weights for better accuracy with small dataset
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to build ResNet50V2 model
def build_resnet50v2_model():
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to build EfficientNetB0 model
def build_efficientnetb0_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train all models
EPOCHS = 10  # Increased epochs for better convergence
INITIAL_EPOCHS = 5

# ConvNeXtTiny
convnext_model = build_convnext_tiny_model()
convnext_history = convnext_model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=val_generator
)
convnext_model.trainable = True  # Fine-tune all layers
convnext_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
convnext_history_fine = convnext_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)
convnext_val_accuracy = convnext_model.evaluate(val_generator)[1]
print(f"ConvNeXtTiny Validation Accuracy: {convnext_val_accuracy:.4f}")

# ResNet50V2
resnet_model = build_resnet50v2_model()
resnet_history = resnet_model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=val_generator
)
resnet_model.trainable = True  # Fine-tune all layers
resnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
resnet_history_fine = resnet_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)
resnet_val_accuracy = resnet_model.evaluate(val_generator)[1]
print(f"ResNet50V2 Validation Accuracy: {resnet_val_accuracy:.4f}")

# EfficientNetB0
efficientnet_model = build_efficientnetb0_model()
efficientnet_history = efficientnet_model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=val_generator
)
efficientnet_model.trainable = True  # Fine-tune all layers
efficientnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
efficientnet_history_fine = efficientnet_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)
efficientnet_val_accuracy = efficientnet_model.evaluate(val_generator)[1]
print(f"EfficientNetB0 Validation Accuracy: {efficientnet_val_accuracy:.4f}")

# Select the best model based on validation accuracy
models = {
    'ConvNeXtTiny': (convnext_model, convnext_val_accuracy),
    'ResNet50V2': (resnet_model, resnet_val_accuracy),
    'EfficientNetB0': (efficientnet_model, efficientnet_val_accuracy)
}
best_model_name = max(models, key=lambda x: models[x][1])
best_model, best_accuracy = models[best_model_name]
print(f"Best Model: {best_model_name} with Validation Accuracy: {best_accuracy:.4f}")

# Save the best model
best_model.save('best_diabetic_retinopathy_model.h5')

# Prediction function
def predict_retinopathy(model, image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0  # Rescale
    prediction = model.predict(img_array)
    predicted_class = tf.argmax(prediction[0]).numpy()
    if predicted_class in [3, 4]:
        return f"Detected: {class_names[predicted_class]} (Class {predicted_class})"
    else:
        return f"Detected: {class_names[predicted_class]} (Class {predicted_class}) - Not Severe/Proliferative"

# Predict on the specified image using the best model
image_path = '16_right_aug_3_aug_3_aug_31_iop.jpg'  # Image in the same directory as train.py
result = predict_retinopathy(best_model, image_path)
print(f"Prediction for 16_right_aug_3_aug_3_aug_31_iop.jpg: {result}")

# Clean up temporary directory
shutil.rmtree(temp_train_dir)