import streamlit as st
import torch
from inference import predict_file
from model import YourModelClass
import os
from urllib.request import urlretrieve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = YourModelClass()
    
    MODEL_URL = "https://huggingface.co/Uniqueorbi/Efficient-net-b5/resolve/main/efficientb5.pth"
    MODEL_PATH = "efficientb5.pth"

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights... This only happens once."):
            urlretrieve(MODEL_URL, MODEL_PATH)

    # Loading the available model file
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

st.title("Music Genre Classifier")
st.markdown("Upload an audio file, and we'll predict the genre using our PyTorch CNN Model!")

uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_path = "temp.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Analyzing audio..."):
        pred = predict_file(temp_path, model, device)
        st.success(f"Prediction: {pred}")
        
    # Optional clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
