# Deepfake Detection System (ResNet152 + Explainability)

A production-ready web application for detecting deepfake videos using a ResNet152-based temporal model, with built-in explainability (visual evidence + interpretable outputs).

---

## 🚀 Overview

This system performs:

- Deepfake classification (Real vs Fake)
- Temporal video analysis (multi-frame sampling)
- Visual evidence extraction
- Explainable AI reporting

Built using:

- PyTorch / Fastai
- FastAPI backend
- HTML + CSS frontend
- OpenCV video processing

---

## 🧠 Model Architecture

- Backbone: **ResNet152**
- Input: 8 sampled frames per video
- Feature extraction per frame
- Temporal aggregation (mean pooling)
- Fully connected classifier

### Output:
- Fake probability
- Real probability
- Final prediction

---

## 🔍 Explainability Features

This system includes:

- Sampled frame visualization
- Temporal evidence panels
- Probability interpretation
- Structured PDF reports

Planned extensions:

- Grad-CAM heatmaps
- SHAP explanations
- LIME explanations

---

## 📁 Project Structure

```text
deepfake_app/
├── main.py
├── requirements.txt
├── render.yaml
├── .gitignore
├── models/
│   └── fusion_detector_v3.pt
├── reports/
├── temp/
├── uploads/
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── app_modules/
    ├── config.py
    ├── model.py
    ├── utils.py
    ├── inference.py
    ├── visual_evidence.py
    └── report_generator.py