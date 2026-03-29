import cv2
import numpy as np
import torch
from torchvision import transforms

from .config import FRAME_COUNT, FRAME_SIZE, NORM_MEAN, NORM_STD


class FFPPInferencePreprocessor:
    def __init__(self, frame_count=FRAME_COUNT):
        self.frame_count = frame_count
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ])

    def load_frames(self, path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        indices = np.linspace(0, max(total - 1, 1), self.frame_count).astype(int)
        frames = []
        times = []

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)
            times.append(i / fps)

        cap.release()
        return frames, times

    def preprocess_video(self, video_path):
        frames, times = self.load_frames(video_path)
        tensor_frames = torch.stack([self.transform(f) for f in frames]).unsqueeze(0)
        return tensor_frames, frames, times


def simple_frame_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    channel_var = float(np.var(frame.reshape(-1, 3), axis=0).mean())

    return {
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge_density,
        "color_variance": channel_var,
    }


def summarise_video_features(frames):
    vals = [simple_frame_features(f) for f in frames]
    summary = {}
    for k in vals[0].keys():
        summary[k] = float(np.mean([v[k] for v in vals]))
    return summary