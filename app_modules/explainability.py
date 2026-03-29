import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from .config import REPORT_DIR, FRAME_SIZE, NORM_MEAN, NORM_STD
from .utils import summarise_video_features


def denormalize(tensor_img):
    x = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * np.array(NORM_STD) + np.array(NORM_MEAN)
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)


def save_sampled_frames_panel(frames, times):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cols = 4
    rows = int(np.ceil(len(frames) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, frame, t in zip(axes, frames, times):
        ax.imshow(frame)
        ax.set_title(f"t={t:.2f}s")
        ax.axis("off")

    for ax in axes[len(frames):]:
        ax.axis("off")

    plt.tight_layout()
    out = REPORT_DIR / "frames_panel.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return str(out)


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # ResNet152 last conv block
        target_layer = self.model.backbone.layer4[-1].conv3
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.activations = out

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, frame_tensor):
        """
        frame_tensor: [1,1,C,H,W]
        """
        self.model.zero_grad()
        logits = self.model(frame_tensor)
        pred_class = torch.argmax(logits, dim=1)
        score = logits[0, pred_class]
        score.backward()

        grads = self.gradients          # [1,C,h,w]
        acts = self.activations         # [1,C,h,w]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (FRAME_SIZE, FRAME_SIZE))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, int(pred_class.item())


def save_gradcam_panel(model, frame_tensor_single, raw_frame):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    gc = GradCAM(model)
    cam, pred_class = gc(frame_tensor_single)

    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(raw_frame, 0.55, heatmap, 0.45, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(raw_frame)
    axes[0].set_title("Representative Frame")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    out = REPORT_DIR / "gradcam_panel.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return str(out)


def save_shap_style_panel(frames):
    """
    This is not exact SHAP on the deep network.
    It is a SHAP-style surrogate explanation using interpretable frame statistics.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    feats = summarise_video_features(frames)
    names = list(feats.keys())
    values = list(feats.values())

    # heuristically convert to signed importance-style display
    centered = np.array(values, dtype=float)
    centered = centered - centered.mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in centered]
    ax.barh(names, centered, color=colors)
    ax.set_title("SHAP-style Surrogate Explanation")
    ax.set_xlabel("Relative contribution (surrogate)")
    plt.tight_layout()

    out = REPORT_DIR / "shap_style_panel.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    return str(out), feats


def build_reason_text(prediction, prob_fake, surrogate_feats):
    reasons = []

    if prediction == "fake":
        reasons.append("The model assigned a higher fake probability after aggregating evidence across sampled frames.")
        if surrogate_feats["edge_density"] > 0.18:
            reasons.append("High edge density suggests unusually sharp local transitions that can be consistent with synthetic blending artefacts.")
        if surrogate_feats["color_variance"] > 1800:
            reasons.append("Elevated colour variance indicates unstable appearance patterns across the clip.")
        if surrogate_feats["contrast"] > 55:
            reasons.append("Higher contrast patterns may reflect unnatural frame-level texture inconsistencies.")
    else:
        reasons.append("The model assigned a higher real probability after averaging evidence across sampled frames.")
        if surrogate_feats["edge_density"] <= 0.18:
            reasons.append("Moderate edge density is more consistent with natural visual structure.")
        if surrogate_feats["color_variance"] <= 1800:
            reasons.append("Stable colour variance suggests more natural appearance consistency across the clip.")
        if surrogate_feats["contrast"] <= 55:
            reasons.append("Contrast levels remain within a range more typical of authentic videos.")

    reasons.append(f"Final fake probability: {prob_fake:.4f}.")
    return reasons