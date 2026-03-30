import torch
from huggingface_hub import hf_hub_download
from .config import DEVICE, HF_REPO_ID, HF_FILENAME, HF_TOKEN
from .model import DeepfakeModel
from .utils import FFPPInferencePreprocessor
from .explainability import (
    save_sampled_frames_panel,
    save_gradcam_panel,
    save_shap_style_panel,
    build_reason_text
)

_model_cache = None
_preprocessor_cache = None


def load_model():
    global _model_cache

    if _model_cache is None:
        # Download model from Hugging Face Hub
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME
                    )
        
        model = DeepfakeModel(pretrained=False).to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        _model_cache = model

    return _model_cache


def get_preprocessor():
    global _preprocessor_cache
    if _preprocessor_cache is None:
        _preprocessor_cache = FFPPInferencePreprocessor()
    return _preprocessor_cache


def predict_video(video_path):
    model = load_model()
    preprocessor = get_preprocessor()

    frames_tensor, raw_frames, times = preprocessor.preprocess_video(video_path)
    frames_tensor = frames_tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(frames_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    real_prob = float(probs[0].cpu().item())
    fake_prob = float(probs[1].cpu().item())
    prediction = "fake" if fake_prob >= 0.5 else "real"

    frames_panel = save_sampled_frames_panel(raw_frames, times)

    # use middle sampled frame for Grad-CAM
    mid_idx = len(raw_frames) // 2
    frame_single = frames_tensor[:, mid_idx:mid_idx+1, :, :, :]
    gradcam_panel = save_gradcam_panel(model, frame_single, raw_frames[mid_idx])

    shap_panel, surrogate_feats = save_shap_style_panel(raw_frames)
    reason_text = build_reason_text(prediction, fake_prob, surrogate_feats)

    return {
        "prediction": prediction,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "reason_text": reason_text,
        "surrogate_features": surrogate_feats,
        "evidence": {
            "frames_panel": frames_panel,
            "gradcam_panel": gradcam_panel,
            "shap_panel": shap_panel
        }
    }
