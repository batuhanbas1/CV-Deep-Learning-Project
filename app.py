"""
YOLOv8n Nesne Tespiti - Streamlit İnteraktif Web Uygulaması
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "best.pt"


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    """Eğitilmiş YOLOv8n modelini önbelleğe alarak yükler."""
    return YOLO(model_path)


def get_device() -> tuple[int | str, str]:
    """CUDA mevcutsa GPU, değilse CPU döndürür."""
    if torch.cuda.is_available():
        return 0, f"CUDA ({torch.cuda.get_device_name(0)})"
    return "cpu", "CPU"


def main() -> None:
    st.set_page_config(
        page_title="YOLOv8n Nesne Tespiti",
        page_icon="🎯",
        layout="wide",
    )

    st.title("🎯 YOLOv8n Gerçek Zamanlı Nesne Tespiti")
    st.markdown(
        "Eğitilmiş **YOLOv8n** (`models/best.pt`) modeli ile yüklediğiniz fotoğraflar "
        "üzerindeki günlük yaşam nesnelerini tespit edin."
    )

    device, device_label = get_device()

    with st.sidebar:
        st.header("⚙️ Model Ayarları")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Nesne tespitinin kabul edilmesi için minimum güven skoru.",
        )
        iou_threshold = st.slider(
            "IoU Threshold (NMS)",
            min_value=0.10,
            max_value=0.95,
            value=0.45,
            step=0.05,
            help="Non-Maximum Suppression için kesişim/birleşim eşiği.",
        )
        st.divider()
        st.subheader("🖥️ Donanım & Model")
        st.caption(f"**Cihaz:** {device_label}")
        st.caption(f"**Ağırlık:** `{MODEL_PATH.relative_to(ROOT_DIR)}`")

    if not MODEL_PATH.exists():
        st.error(
            f"Model dosyası bulunamadı: `{MODEL_PATH}`. "
            "Lütfen önce `python train.py` komutunu çalıştırın."
        )
        st.stop()

    model = load_model(str(MODEL_PATH))

    uploaded_file = st.file_uploader(
        "Bir fotoğraf yükleyin (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Analiz başlatmak için yukarıdan bir fotoğraf seçin.")
        return

    pil_image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(pil_image)

    t0 = time.perf_counter()
    results = model.predict(
        source=img_rgb,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        device=device,
        verbose=False,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0

    res = results[0]
    inference_ms = (
        float(res.speed.get("inference", wall_ms))
        if hasattr(res, "speed")
        else wall_ms
    )

    annotated_bgr = res.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Orijinal Fotoğraf")
        st.image(pil_image, use_container_width=True)
    with col2:
        st.subheader("🔍 Tespit Çıktısı")
        st.image(annotated_rgb, use_container_width=True)

    st.divider()

    detections = []
    if res.boxes is not None and len(res.boxes) > 0:
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = res.names.get(cls_id, str(cls_id))
            conf_pct = round(float(box.conf[0].item()) * 100.0, 2)
            x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "Sınıf Adı": cls_name,
                    "Güven Skoru (%)": conf_pct,
                    "Koordinatlar (x1, y1, x2, y2)": f"[{x1}, {y1}, {x2}, {y2}]",
                }
            )

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(label="⚡ Inference Time", value=f"{inference_ms:.2f} ms")
    with m_col2:
        st.metric(label="📦 Tespit Edilen Nesne Sayısı", value=len(detections))

    st.subheader("📋 Tespit Edilen Nesneler Tablosu")
    if detections:
        df = pd.DataFrame(detections)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("Seçilen eşik değerleriyle nesne tespit edilemedi.")


if __name__ == "__main__":
    main()
