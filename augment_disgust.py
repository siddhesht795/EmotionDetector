# augment_disgust.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

def augment_disgust_class(source_dir, target_count=4000, image_size=(48, 48)):
    save_dir = source_dir  # Augmented images saved in same folder

    # Create directory if not exists
    os.makedirs(save_dir, exist_ok=True)

    # Data augmentation settings
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    existing_images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.png'))]
    num_existing = len(existing_images)
    num_needed = target_count - num_existing
    augmented_count = 0

    for img_name in existing_images:
        img_path = os.path.join(source_dir, img_name)
        img = load_img(img_path, color_mode='grayscale', target_size=image_size)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        prefix = os.path.splitext(img_name)[0]
        for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix=prefix + "_aug", save_format='jpg'):
            augmented_count += 1
            if augmented_count >= num_needed:
                break
        if augmented_count >= num_needed:
            break

    print(f"Augmentation complete: {augmented_count} new images generated for 'disgust'.")

if __name__ == "__main__":
    # Change path as needed
    augment_disgust_class("../train/disgust")
