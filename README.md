# 🎙️ Emotion Recognition from Speech | CodeAlpha Internship Project

This project is part of the **CodeAlpha Machine Learning Internship**. It focuses on recognizing human emotions such as happy, sad, angry, calm, etc., from speech audio using machine learning and deep learning techniques.

---

## 🎯 Objective

Build a model that classifies human emotions based on voice signals using MFCC features and a deep learning neural network.

---

## 📁 Dataset

- **Dataset Used:** [RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song](https://zenodo.org/record/1188976)
- **Emotions Covered:**  
  `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- joblib

---

## ⚙️ Features Used

- **MFCC (Mel-Frequency Cepstral Coefficients):** Audio feature used to represent short-term power spectrum of sound.

---

## 📊 Model Workflow

1. **Data Preprocessing:** Load audio files, extract MFCCs.
2. **Feature Extraction:** Take average of MFCCs across time.
3. **Modeling:** Train a neural network classifier.
4. **Saving:** Save the model (`.h5`) and label encoder (`.pkl`).
5. **Prediction:** Predict emotion from any given `.wav` file.

---

## 📁 Files Included

| File/Folder               | Description                                 |
|--------------------------|---------------------------------------------|
| `Emotion_Recognition.ipynb` | Jupyter notebook with full pipeline         |
| `final_emotion_model.h5` | Trained Keras model                         |
| `label_encoder.pkl`      | Saved label encoder using `joblib`          |
| `requirements.txt`       | Python package dependencies                 |
| `README.md`              | Project description                         |

---

## 🔮 Prediction Example

```python
# Load saved model and encoder
from tensorflow.keras.models import load_model
import joblib

model = load_model("final_emotion_model.h5")
label_encoder = joblib.load("label_encoder.pkl")

# Predict function
def predict_emotion(audio_path):
    import librosa
    import numpy as np
    audio, sr = librosa.load(audio_path, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0).reshape(1, -1)
    prediction = model.predict(mfcc_scaled)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    print("✅ Predicted Emotion:", predicted_label[0])
```

---

## 🚀 How to Run the Project

1. Clone the repo:
```bash
git clone https://github.com/Abbas004tech/CodeAlpha_Emotion_Recognition_From_Speech.git
cd CodeAlpha_Emotion_Recognition_From_Speech
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch notebook:
```bash
jupyter notebook
```

4. Use the prediction function with any `.wav` audio file.

---

## ✅ Project Results

- **Model Accuracy:** ~84%
- **Successfully Recognized Emotions:** happy, angry, calm, fearful, neutral, sad

---

## 📦 requirements.txt

```txt
numpy
pandas
matplotlib
scikit-learn
tensorflow
librosa
soundfile
joblib
```

---

## 👤 Author

**Abbas004tech**  
🔗 [GitHub Profile](https://github.com/Abbas004tech)

---

