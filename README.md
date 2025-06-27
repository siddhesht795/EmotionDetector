# 🧠 Emotion Detection using CNN

This project uses a **Convolutional Neural Network (CNN)** to classify facial expressions into seven emotions using grayscale images of size **48x48**. It’s built entirely from scratch with TensorFlow/Keras, designed for learning and experimentation with image classification, augmentation, and performance metrics.

---

## 📸 Sample Outputs

| Original Image | Predicted Emotion |
|----------------|-------------------|
| ![img1](assets/sample_face1.png) | 😄 Happy |
| ![img2](assets/sample_face2.png) | 😢 Sad  |
| ![img3](assets/sample_face3.png) | 😡 Angry |

> 📁 *Put your sample prediction results inside an `assets/` folder to show how your model performs visually.*

---

## 📂 Dataset Structure

The dataset is a folder-based image classification dataset with the following structure:

```
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── same as above
```

- 📏 **Image Size**: 48x48 pixels  
- 🌈 **Color Mode**: Grayscale  
- 🧠 **Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

---

## 🔧 Techniques & Features Used

- **CNN Architecture** built from scratch
- **Image Augmentation** via `ImageDataGenerator`
- **Dropout** for regularization
- **Batch Normalization** to improve stability
- **Class Weights** to handle imbalanced datasets
- **Callbacks**: 
  - `EarlyStopping`
  - `ReduceLROnPlateau`
- **Evaluation Metrics**: 
  - Accuracy
  - Precision
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)

---

## 🏗️ Model Architecture

```python
Conv2D → BatchNorm → MaxPooling → Dropout
→ Conv2D → BatchNorm → MaxPooling → Dropout
→ Conv2D → BatchNorm → MaxPooling → Dropout
→ Flatten → Dense(512) → Dropout
→ Dense(256) → Dropout
→ Dense(7, softmax)
```

---

## 📈 Performance Summary

### 🧾 Confusion Matrix

![confusion-matrix](assets/confusion_matrix.png)

> Make sure to place your actual confusion matrix image in `assets/confusion_matrix.png`.

### 🧪 Sample Output Metrics

```
              precision    recall  f1-score   support

       angry       0.46      0.04      0.07       956
     disgust       1.00      0.01      0.02       111
        fear       0.26      0.09      0.14      1024
       happy       0.67      0.88      0.76      1774
     neutral       0.41      0.60      0.49      1233
         sad       0.35      0.49      0.41      1247
    surprise       0.67      0.69      0.68       831

    accuracy                           0.50      7176
   macro avg       0.55      0.40      0.37      7176
weighted avg       0.49      0.50      0.45      7176
```

---

## 🚀 How to Run

### 🖥️ Requirements

- Python 3.8+
- TensorFlow
- Matplotlib
- Seaborn
- NumPy
- scikit-learn

Install everything using:
```bash
pip install -r requirements.txt
```

### 🧪 Train the Model

```bash
python train_model.py
```

- This will:
  - Train the CNN on your data
  - Use `ReduceLROnPlateau` and `EarlyStopping`
  - Plot confusion matrix
  - Save the model as `emotion_model.h5`

---

## 💾 Inference on New Images

You can test predictions on new images using a separate script (you can create one named `predict_image.py`):

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('emotion_model.h5')
img = image.load_img("path/to/your/image.jpg", color_mode='grayscale', target_size=(48, 48))
img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
img_array = np.expand_dims(img_array, -1)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
print("Predicted emotion:", predicted_class)
```

---

## 📉 Challenges & Learnings

- Emotion detection is **very imbalanced** — some classes like `disgust` had <500 images.
- Handling such imbalances using **class weights** helped reduce bias.
- Adding **data augmentation** and **dropout** improved generalization.
- Realized that accuracy isn’t everything — precision, recall, and confusion matrix matter!

---

## 💡 Future Improvements

- Integrate **Transfer Learning** (MobileNet / EfficientNet)
- Use **RGB images** with 224x224 size
- Deploy using **Streamlit** or **Flask**
- Enable **real-time webcam predictions**

---

## 🧠 Why This Project?

> This project was done as a **learning experience** to understand CNNs, model evaluation, overfitting handling, and how small tweaks in architecture and data processing can significantly affect performance.

---

## 📜 License

This project is open-source and free to use under [MIT License](LICENSE).

---

## 🧳 Author

**Siddhesh Todi**  
Feel free to connect if you’re exploring similar projects!
