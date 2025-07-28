import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np


# set base directory for the dataset
base_dir = 'Q2' 

# define paths for training and testing directories
train_directory = os.path.join(base_dir, 'train')
test_directory = os.path.join(base_dir, 'test')

# set image dimensions and batch size
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# create training data generator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,              # normalize pixel values to 0-1
    rotation_range=40,           # randomly rotate images
    width_shift_range=0.2,       # randomly shift images horizontally
    height_shift_range=0.2,      # randomly shift images vertically
    shear_range=0.2,             # shear transformation
    zoom_range=0.2,              # randomly zoom
    horizontal_flip=True,        # flip images horizontally
    fill_mode='nearest'          # fill in missing pixels
)

# create a test data generator (no augmentation, just rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training images from directory
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'  # binary labels (0 for cat, 1 for dog)
)

# load validation/test images from directory
validation_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# create a Sequential model
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    # this is the first convolutional layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(2, 2),

    # this is the second convolutional layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # this is the third convolutional layer
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # flatten the output from the convolutional layers
    layers.Flatten(),
    layers.Dropout(0.5),  # dropout layer to prevent overfitting

    # connect to a fully connected layer
    layers.Dense(512, activation='relu'),

    # output layer: 1 neuron with sigmoid activation for binary classification
    layers.Dense(1, activation='sigmoid')
])

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# history store the training process results (loss and accuracy per epoch)
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=35,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# save the model to a file
model.save('Q2/cat_dog_model.h5') 

# evaluate the model on the test set
loss, accuracy = model.evaluate(validation_generator)
print(f"Test Accuracy: {accuracy:.4f}")

# Load the saved model
model = load_model('Q2/cat_dog_model.h5')

# Path to an internet image 
img_path = 'Q2/chihuahua.webp' 
#img_path = 'Q2/golden-retriever.jpg'
#img_path = 'Q2/maine_coon.webp'

# Load and preprocess image
img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Match training preprocessing

# Make prediction
prediction = model.predict(img_array)[0][0]
label = 'Dog' if prediction > 0.5 else 'Cat'
confidence = prediction if prediction > 0.5 else 1 - prediction

# Show prediction
print(f"Prediction: {label} (Confidence: {confidence:.2f})")

# Optional: Display the image with the label
plt.imshow(img)
plt.title(f"Prediction: {label} ({confidence:.2f})")
plt.axis('off')
plt.show()