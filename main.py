import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app_modules.config import (
    STATIC_DIR, TEMPLATES_DIR, TEMP_DIR,
    ALLOWED_EXTENSIONS, MAX_UPLOAD_MB
)
from app_modules.inference import predict_video
from app_modules.report_generator import generate_pdf_report

app = FastAPI(title="Deepfake Detector")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def validate_upload(file: UploadFile):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    content_length = file.headers.get("content-length")
    if content_length:
        size_mb = int(content_length) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            raise HTTPException(status_code=400, detail=f"File too large. Max size is {MAX_UPLOAD_MB} MB.")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    temp_path = None
    try:
        validate_upload(file)

        suffix = Path(file.filename).suffix.lower()
        temp_path = TEMP_DIR / f"upload{suffix}"

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_video(str(temp_path))
        generate_pdf_report(file.filename, result)

        return JSONResponse({
            "prediction": result["prediction"],
            "fake_probability": result["fake_probability"],
            "reason_text": result["reason_text"],
            "frames_panel_url": "/reports/frames_panel.png",
            "gradcam_panel_url": "/reports/gradcam_panel.png",
            "shap_panel_url": "/reports/shap_style_panel.png",
            "report_url": "/report/download"
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Inference failed: {str(e)}"}
        )
    finally:
        try:
            if temp_path and temp_path.exists():
                os.remove(temp_path)
        except Exception:
            pass


@app.get("/report/download")
async def download_report():
    report_path = Path("reports/deepfake_report.pdf")
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")

    return FileResponse(
        str(report_path),
        media_type="application/pdf",
        filename="deepfake_report.pdf"
    )