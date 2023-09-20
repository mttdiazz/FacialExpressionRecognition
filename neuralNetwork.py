import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Define the paths to your training and testing data directories
trainingDataDir = '/Users/mateo/Documents/Mateo/FacialExpressionRecognition/FacialExpression/train'
testingDataDir = '/Users/mateo/Documents/Mateo/FacialExpressionRecognition/FacialExpression/test'

# Set the input image dimensions and other hyperparameters
img_width, img_height = 48, 48
batch_size = 64
epochs = 50

# Create data generators for loading and augmenting the training and testing data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load and augment the training data
train_generator = train_datagen.flow_from_directory(
    trainingDataDir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    subset='training',
    shuffle=True
)

# Load and augment the validation data
validation_generator = train_datagen.flow_from_directory(
    trainingDataDir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    subset='validation',
    shuffle=True
)

# Define the upgraded model architecture
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile the upgraded model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the upgraded model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the upgraded model
model.save('FacialExpressionModel_Upgraded.h5')

# Evaluate the upgraded model on the testing data
test_generator = test_datagen.flow_from_directory(
    testingDataDir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator)
print('Testing Loss:', loss)
print('Testing Accuracy:', accuracy)