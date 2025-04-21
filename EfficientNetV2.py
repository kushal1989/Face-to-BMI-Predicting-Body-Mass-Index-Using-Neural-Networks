import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Data Paths
train_dir = r'C:\PROJECT\Dataset\Model2\train'
val_dir = r'C:\PROJECT\Dataset\Model2\test'
csv_file = r'C:\PROJECT\Dataset\Model2\BMI(kg m).csv'

# Load and clean data
data = pd.read_csv(csv_file)

# Check for NaN values in BMI and handle them
if data['bmi'].isna().sum() > 0:
    print(f"Found {data['bmi'].isna().sum()} NaN values in the BMI column. Replacing with mean.")
    data['bmi'] = data['bmi'].fillna(data['bmi'].mean())  # Replace NaN with mean value

# Normalize BMI values
bmi_mean = data['bmi'].mean()
bmi_std = data['bmi'].std()
data['bmi'] = (data['bmi'] - bmi_mean) / bmi_std

# Reduce dataset size for faster training
train_labels = data['bmi'].values[:10000]  # First 10,000 samples for training
val_labels = data['bmi'].values[10000:15000]  # Next 5,000 samples for validation

# Check for NaN values after processing
assert not np.any(np.isnan(train_labels)), "Training labels contain NaN values!"
assert not np.any(np.isnan(val_labels)), "Validation labels contain NaN values!"

# Data generators with normalization and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Define data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Reduced image size
    batch_size=64,          # Larger batch size for faster training
    class_mode=None,
    shuffle=False
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=64,
    class_mode=None,
    shuffle=False
)

# Custom generator to align images and labels
def custom_data_generator(image_generator, labels):
    while True:
        for batch_images in image_generator:
            idx = image_generator.index_array[:len(batch_images)]
            yield batch_images, labels[idx]

# Create training and validation data generators
train_data_gen = custom_data_generator(train_generator, train_labels)
val_data_gen = custom_data_generator(validation_generator, val_labels)

# Define the EfficientNetV2 model
base_model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Unfreeze the top layers for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Add custom layers for BMI prediction
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='linear')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=1e-5), loss='mean_squared_error', metrics=['mae'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint(filepath="checkpoint_efficientnetv2.keras", save_best_only=True, monitor="val_loss", verbose=1)

# Train the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=len(train_labels) // 64,
    epochs=20,  # Reduced number of epochs for faster training
    validation_data=val_data_gen,
    validation_steps=len(val_labels) // 64,
    callbacks=[early_stopping, lr_scheduler, checkpoint]
)

# Save the final model
model.save("Optimized_EfficientNetV2_Model.keras")
print("Final model saved.")
