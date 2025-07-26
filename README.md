# 🎧 Emotion Recognition from Speech - CodeAlpha Project

This repository contains the implementation of an **Emotion Recognition System** using speech audio. This project was completed as part of the **CodeAlpha Machine Learning Internship**.

---

## 🔍 Objective

To classify human emotions such as **happy**, **angry**, **sad**, **calm**, **fearful**, and **neutral** from audio recordings using machine learning and deep learning techniques.

---

## 🗂️ Dataset

- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
- Audio data from 24 professional actors vocalizing statements with various emotions.
- Emotions covered: *calm, happy, sad, angry, fearful, neutral*

---

## 🛠️ Technologies Used

- **Python**
- **Librosa** – audio processing and MFCC extraction
- **TensorFlow / Keras** – deep learning model
- **Scikit-learn** – evaluation metrics
- **Matplotlib & Seaborn** – data visualization
- **NumPy & Pandas** – data handling

---

## 🎯 Features Extracted

- **MFCC (Mel-Frequency Cepstral Coefficients)** – 40 coefficients per sample
- Each audio clip is converted into a numerical feature vector for training

---

## 🤖 Model Architecture

- Deep Learning model using **LSTM layers**
- Input: Scaled MFCC features
- Output: Emotion class
- Trained with categorical cross-entropy loss and Adam optimizer

---

## 📊 Results

- **Model Accuracy**: ~84%
- Evaluation metrics include precision, recall, and F1-score for each emotion class

---

## 🧪 How to Test

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

🌐 References

RAVDESS Dataset: https://zenodo.org/record/1188976
Librosa Documentation: https://librosa.org/
TensorFlow: https://www.tensorflow.org/


⭐ If you found this project useful, feel free to star the repo and share it!
