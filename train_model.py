import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

train_dir = "./train"
test_dir = "./test"

image_size = (48, 48)

train_datagen = ImageDataGenerator(
    rescale=1./255,               # Always normalize
    rotation_range=30,            # Rotate images up to 30°
    zoom_range=0.2,               # Zoom in by 20%
    width_shift_range=0.2,        # Shift image left/right by 20%
    height_shift_range=0.2,       # Shift image up/down by 20%
    horizontal_flip=True          # Flip image horizontally
)

test_datagen = ImageDataGenerator(
    rescale=1./255,               # Always normalize
    rotation_range=30,            # Rotate images up to 30°
    zoom_range=0.2,               # Zoom in by 20%
    width_shift_range=0.2,        # Shift image left/right by 20%
    height_shift_range=0.2,       # Shift image up/down by 20%
    horizontal_flip=True          # Flip image horizontally
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (48, 48),
    color_mode = 'grayscale',
    batch_size = 64,
    class_mode = 'categorical',
    shuffle = True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (48, 48),
    color_mode = 'grayscale',
    batch_size = 64,
    class_mode = 'categorical',
    shuffle = False
)

print(train_generator.class_indices)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation = "softmax"))

model.compile(
    loss='categorical_crossentropy',  # Good for multi-class classification
    optimizer='adam',                 # Fast and reliable optimizer
    metrics=['accuracy']              # To track performance
)

loss, accuracy = model.evaluate(test_generator)
print("Test accuracy:", accuracy)


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

model.save('emotion_model.h5')
