# Music Genre Classification

Upload an audio file → predicts genre using PyTorch's EfficientNet B5

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment on Hugging Face Spaces

1. Go to [https://huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create Space**
3. Select SDK → **Streamlit**, and Visibility → Public
4. Connect this GitHub repository or manually upload the files including `.pth` model file.
5. Hugging Face will automatically install dependencies from `requirements.txt` and launch the app.
