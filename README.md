# 🧵 Pattern Sense - Fabric Pattern Classifier

Pattern Sense is a deep learning-powered web application built using **TensorFlow** and **Streamlit**. It allows users to upload an image of a fabric and accurately predicts its pattern type from one of 10 classes such as Checks, Dotted, Floral, Striped, and more.

---

## 🚀 Features

- 🖼️ Upload fabric images in JPG, JPEG, or PNG
- 🧠 Real-time pattern prediction using a CNN model
- 📊 Displays confidence score and class label
- 💅 Clean, modern Streamlit UI
- 🧵 Supports 10 fabric pattern classes:
  - `Checks`, `Dotted`, `Floral`, `Geometric`, `Herringbone`, `Ikat`, `Paisley`, `Plain`, `Printed`, `Striped`

---

## 🏗️ Tech Stack

- **Frontend**: Streamlit (with custom CSS)
- **Model**: TensorFlow / Keras
- **Backend**: Python
- **Image Processing**: Pillow, OpenCV
- **Visualization**: Matplotlib, Seaborn

---

## 🧪 Model Training

The model is trained using a custom CNN with:
- 4 Conv2D + MaxPooling layers
- Dropout and Dense layers
- Categorical Crossentropy Loss
- `ImageDataGenerator` for real-time data augmentation

**Data:**  
TFD Textile Dataset stored at:
