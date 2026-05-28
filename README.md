# 🎙️ Speech Emotion Recognition (SER) using CNN

A deep learning project that classifies human emotions from speech audio using a 1D Convolutional Neural Network trained on the [RAVDESS](https://zenodo.org/record/1188976) dataset.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **91%** |
| Macro F1-Score | **0.91** |
| Epochs Trained | 40 (early stopping) |

### Per-Class Performance

| Emotion   | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Neutral   | 0.96      | 0.82   | 0.89     |
| Calm      | 0.89      | 0.98   | 0.93     |
| Happy     | 0.92      | 0.93   | 0.92     |
| Sad       | 0.80      | 0.92   | 0.85     |
| Angry     | 0.93      | 0.96   | 0.94     |
| Fearful   | 0.92      | 0.90   | 0.91     |
| Disgust   | 0.95      | 0.91   | 0.93     |
| Surprised | 0.94      | 0.84   | 0.89     |


## 🏗️ Model Architecture

```
Input: (40, 1)  ← 40 MFCC features
  │
  ├─ Conv1D(128, kernel=5) → BatchNorm → ReLU → Dropout(0.3)
  ├─ Conv1D(256, kernel=5) → BatchNorm → ReLU → Dropout(0.3)
  ├─ Flatten
  ├─ Dense(256) → ReLU → Dropout(0.3)
  ├─ Dense(128) → ReLU
  └─ Dense(8)   → Softmax
```

---

## 📁 Project Structure

```
SER/
├── SER.ipynb               # Main notebook (training + evaluation)
├── requirements.txt        # Python dependencies
├── assets/
│   ├── loss_curve.png
│   └── accuracy_curve.png
├── dataset/                # RAVDESS dataset (not tracked by git)
│   └── Actor_XX/
│       └── *.wav
└── saved_model/            # Saved scaler + data (not tracked by git)
    ├── x.joblib
    └── y.joblib
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Create a virtual environment

```bash
python -m venv myenv
# Windows
myenv\Scripts\activate
# macOS/Linux
source myenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the [RAVDESS dataset](https://zenodo.org/record/1188976) and place it in a `dataset/` folder:

```
dataset/
├── Actor_01/
├── Actor_02/
...
└── Actor_24/
```

### 5. Run the notebook

```bash
jupyter notebook SER.ipynb
```

Update the `DATASET_PATH` variable in the notebook to point to your `dataset/` folder.

---

## 🎤 Predict on Your Own Audio

```python
# Record a 7-second clip and predict
predict_emotion("path/to/your/audio.wav")
```

Or use the live microphone recording cell in the notebook.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** — CNN model
- **librosa** — audio feature extraction (MFCC)
- **scikit-learn** — preprocessing, evaluation
- **sounddevice** — microphone recording
- **NumPy / Matplotlib**

---

## 📌 Key Design Choices

- **MFCC features** (40 coefficients, mean-pooled across time frames)
- **Silence trimming** with `librosa.effects.trim` before feature extraction
- **StandardScaler** normalization on features
- **EarlyStopping** with patience=10 on `val_loss`
- **Adam optimizer** with lr=0.001

---

## 📜 Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song  
24 actors, 8 emotions, 2452 audio files.  
[https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)

---

## 📄 License

MIT License
