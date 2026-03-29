from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from .config import REPORT_DIR


def generate_pdf_report(filename, result):
    pdf_path = REPORT_DIR / "deepfake_report.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4

    y = height - 40
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Deepfake Detection Report")

    y -= 28
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"File: {filename}")

    y -= 16
    c.drawString(40, y, f"Prediction: {result['prediction']}")

    y -= 16
    c.drawString(40, y, f"Fake probability: {result['fake_probability']:.4f}")

    y -= 16
    c.drawString(40, y, f"Real probability: {result['real_probability']:.4f}")

    y -= 24
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Why this prediction?")

    c.setFont("Helvetica", 9)
    for line in result["reason_text"]:
        y -= 14
        c.drawString(50, y, f"- {line}")
        if y < 120:
            c.showPage()
            y = height - 40

    panels = [
        ("Sampled Frame Evidence", result["evidence"]["frames_panel"]),
        ("Grad-CAM Explanation", result["evidence"]["gradcam_panel"]),
        ("SHAP-style Surrogate Explanation", result["evidence"]["shap_panel"]),
    ]

    for title, panel in panels:
        c.showPage()
        y = height - 40
        c.setFont("Helvetica-Bold", 13)
        c.drawString(40, y, title)
        y -= 20

        p = Path(panel)
        if p.exists():
            c.drawImage(ImageReader(str(p)), 30, y - 430, width=535, height=400)

    c.save()
    return str(pdf_path)