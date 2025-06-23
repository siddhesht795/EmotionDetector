import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('emotion_model.h5')

class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

img = image.load_img('"E:/Siddhesh/ML/EmotionDetector/github/images/test_img2.jpg"', target_size = (48, 48), color_mode = 'grayscale')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
predicted_class = class_labels[np.argmax(prediction)]

print("Predicted Emotion:", predicted_class)
print("Prediction Probabilities:", prediction)
