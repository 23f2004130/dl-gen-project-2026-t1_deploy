import librosa
import numpy as np
import torch
import torchvision.transforms as T

# Standard GTZAN mapping as a fallback, modify as needed
GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

def predict_file(path, model, device):
    """
    Process the audio file and return the predicted genre.
    """
    y, _ = librosa.load(path, sr=32000, mono=True)

    stride = 32000 * 2
    clip_sample = 32000 * 5

    logits_list = []

    for start in range(0, max(1, len(y) - clip_sample + 1), stride):
        clip = y[start:start + clip_sample]

        if len(clip) < clip_sample:
            clip = np.pad(clip, (0, clip_sample - len(clip)))

        # Extracted Hardcoded globals as per instructions, local directly inside function
        mel = librosa.feature.melspectrogram(
            y=clip,
            sr=32000,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )

        logmel = librosa.power_to_db(mel, ref=np.max)
        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)

        logmel = np.stack([logmel]*3, axis=0) # Make it 3 channel
        logmel = torch.tensor(logmel, dtype=torch.float32)

        logmel = T.Resize((456,456))(logmel)
        logmel = logmel.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(logmel)
            logits_list.append(logits.cpu())

    if len(logits_list) == 0:
        return "Unknown - Audio too short"

    # Adding Top-3 predictions and confidence scores
    probs = torch.nn.functional.softmax(avg_logits, dim=1)[0]
    top3_prob, top3_idx = torch.topk(probs, 3)

    results = []
    for i in range(3):
        idx = top3_idx[i].item()
        confidence= top3_prob[i].item() * 100
        genre_name = GENRES[idx] if idx < len(GENRES) else f"Class {idx}"
        results.append(f"{genre_name}: {confidence:.1f}%")
        
    return " | ".join(results)
