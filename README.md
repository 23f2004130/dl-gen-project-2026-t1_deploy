---

title: Music Genre Classifier
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.33.0"
python_version: "3.10"
app_file: app.py
pinned: false
-------------

# Music Genre Classification (Deep Learning)

A production-ready music genre classification system built using deep learning on Mel Spectrogram representations. The model processes raw audio files, converts them into log-Mel spectrograms, and predicts the most likely genre.

---

## Demo

Hugging Face Space: *https://huggingface.co/spaces/Uniqueorbi/music-genre-classifier*

Upload an audio file (`.wav` / `.mp3`) and get genre predictions instantly.

---

## Model Overview

* Architecture: CNN / PANNs-based audio classifier
* Input: Log-Mel Spectrogram (3-channel)
* Output: Genre probabilities
* Inference: Sliding window over audio clips + aggregation

---

## Pipeline

### 1. Audio Processing

* Resampling to fixed sample rate
* Segmentation into fixed-length clips
* Padding for short clips

### 2. Feature Extraction

* Mel Spectrogram generation
* Log scaling (dB)
* Normalization
* Channel expansion (3-channel input)
* Resize to model input resolution

### 3. Inference

* Model prediction on each segment
* Softmax probability computation
* Aggregation across segments
* Final genre selection (argmax)

---

## Project Structure

```
music-genre-classifier/
│
├── app.py                # Streamlit UI
├── model.py              # Model architecture
├── inference.py          # Prediction pipeline
├── utils.py              # Audio preprocessing
├── requirements.txt
├── README.md
```

---

## Model Weights

Due to GitHub file size limits, model weights are hosted externally.

* Automatically downloaded at runtime from Hugging Face
* No manual setup required

---

## Installation (Local)

```bash
git clone https://github.com/YOUR_USERNAME/music-genre-classifier.git
cd music-genre-classifier
pip install -r requirements.txt
```

---

## Run Locally

```bash
streamlit run app.py
```
