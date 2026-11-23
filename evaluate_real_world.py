"""
test_images/raw/ klasöründeki gerçek dünya fotoğraflarını models/best.pt ile
değerlendirir, kutucuklu çıktıları test_images/outputs/ klasörüne kaydeder ve
test_images/real_world_report.md karşılaştırma raporunu üretir.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def evaluate_real_world(
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> None:
    root = Path(__file__).resolve().parent
    raw_dir = root / "test_images" / "raw"
    out_dir = root / "test_images" / "outputs"
    model_path = root / "models" / "best.pt"
    report_path = root / "test_images" / "real_world_report.md"

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Eğitilmiş model bulunamadı: {model_path}. Önce train.py çalıştırın."
        )

    image_paths = sorted(
        p for p in raw_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(
            f"[WARN] {raw_dir} içinde değerlendirilecek fotoğraf bulunamadı. "
            "Lütfen telefonla çektiğiniz 5 fotoğrafı bu klasöre koyup tekrar çalıştırın."
        )
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))

    # Warm-up
    model.predict(
        source=str(image_paths[0]),
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        device=device,
        verbose=False,
    )

    rows: list[str] = []
    total_detections = 0
    inference_times: list[float] = []

    for img_path in image_paths:
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            device=device,
            verbose=False,
        )
        res = results[0]
        inf_ms = float(res.speed.get("inference", 0.0)) if hasattr(res, "speed") else 0.0
        inference_times.append(inf_ms)

        annotated_bgr = res.plot()
        out_file = out_dir / img_path.name
        cv2.imwrite(str(out_file), annotated_bgr)

        detected_items: list[str] = []
        conf_items: list[str] = []
        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item()) * 100.0
                cls_name = res.names.get(cls_id, str(cls_id))
                detected_items.append(cls_name)
                conf_items.append(f"{cls_name}: %{conf:.1f}")
            total_detections += len(res.boxes)

        objects_str = ", ".join(detected_items) if detected_items else "Tespit Yok"
        confs_str = ", ".join(conf_items) if conf_items else "-"
        rows.append(
            f"| `{img_path.name}` | {objects_str} | {confs_str} | {inf_ms:.2f} ms |"
        )
        print(f"[INFO] İşlendi: {img_path.name} -> {out_file.name} ({inf_ms:.2f} ms)")

    avg_inf = sum(inference_times) / len(inference_times) if inference_times else 0.0
    report_lines = [
        "# Gerçek Dünya Fotoğrafları Değerlendirme Raporu",
        "",
        f"- **Model:** `models/best.pt`",
        f"- **Confidence Threshold:** `{conf_threshold}`",
        f"- **IoU Threshold:** `{iou_threshold}`",
        f"- **Toplam Fotoğraf:** `{len(image_paths)}`",
        f"- **Toplam Tespit Sayısı:** `{total_detections}`",
        f"- **Ortalama Inference Süresi:** `{avg_inf:.2f} ms/image`",
        "",
        "| Fotoğraf | Tespit Edilen Nesneler | Güven Skorları (%) | Inference Süresi (ms) |",
        "| :--- | :--- | :--- | :---: |",
        *rows,
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[INFO] Rapor kaydedildi: {report_path}")


if __name__ == "__main__":
    evaluate_real_world()
