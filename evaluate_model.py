import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model("models/emotion_model.h5")

test_dir = "../test"
image_size = (48, 48)
batch_size = 64

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Prediction
predictions = model.predict(test_generator, verbose=1)
predicted_labels = np.argmax(predictions, axis=1)

print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_labels))

cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

result = model.evaluate(test_generator, verbose=1)
print(f"\nTest Loss: {result[0]:.4f}")
print(f"Test Accuracy: {result[1]:.4f}")
