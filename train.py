"""
YOLOv8n modelini özel COCO128 (Train/Val/Test) veri seti üzerinde eğitir,
4GB VRAM için optimize edilmiş CUDA ayarlarıyla (CPU fallback destekli) çalıştırır,
eğitim süresini ve test metriklerini (mAP@50, mAP@50-95, Precision, Recall,
Ortalama IoU, F1-Score, Inference Time) ölçerek results/metrics.json ve
models/best.pt çıktılarını üretir.
"""

from __future__ import annotations

import json
import platform
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from prepare_data import prepare_dataset

EPOCHS = 25
IMGSZ = 640
BATCH_SIZE = 16
OPTIMIZER = "AdamW"
LR0 = 0.001
WEIGHT_DECAY = 0.0005
WORKERS = 2
SEED = 42


def get_device() -> tuple[int | str, str]:
    """CUDA mevcutsa GPU 0, değilse CPU döndürür."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return 0, f"CUDA ({gpu_name})"
    return "cpu", "CPU"


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """İki [N, 4] ve [M, 4] xyxy kutu dizisi arasındaki IoU matrisini hesaplar."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    ax1, ay1, ax2, ay2 = np.split(boxes_a, 4, axis=1)
    bx1, by1, bx2, by2 = np.split(boxes_b, 4, axis=1)

    inter_x1 = np.maximum(ax1, bx1.T)
    inter_y1 = np.maximum(ay1, by1.T)
    inter_x2 = np.minimum(ax2, bx2.T)
    inter_y2 = np.minimum(ay2, by2.T)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a + area_b.T - inter_area

    return np.where(union > 0, inter_area / union, 0.0)


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """YOLO formatındaki etiket dosyasını (cls, xyxy) piksel koordinatlarına çevirir."""
    if not label_path.exists():
        return np.empty((0,), dtype=int), np.empty((0, 4), dtype=np.float32)

    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return np.empty((0,), dtype=int), np.empty((0, 4), dtype=np.float32)

    classes = []
    boxes = []
    for line in lines:
        parts = line.split()
        cls_id = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        x1 = (cx - w / 2.0) * img_w
        y1 = (cy - h / 2.0) * img_h
        x2 = (cx + w / 2.0) * img_w
        y2 = (cy + h / 2.0) * img_h
        classes.append(cls_id)
        boxes.append([x1, y1, x2, y2])

    return np.array(classes, dtype=int), np.array(boxes, dtype=np.float32)


def evaluate_test_iou_and_speed(
    model: YOLO,
    test_images_dir: Path,
    test_labels_dir: Path,
    device: int | str,
) -> tuple[float, float, int, int]:
    """
    Test görüntüleri üzerinde ortalama IoU ve tekil görüntü ortalama
    inference süresini (ms/image) ölçer.
    """
    image_paths = sorted(
        p for p in test_images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        return 0.0, 0.0, 0, 0

    # CUDA warm-up
    model.predict(
        source=str(image_paths[0]),
        imgsz=IMGSZ,
        conf=0.25,
        iou=0.45,
        device=device,
        verbose=False,
    )

    matched_ious: list[float] = []
    inference_times_ms: list[float] = []
    total_gt_boxes = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        gt_cls, gt_boxes = load_gt_boxes(
            test_labels_dir / f"{img_path.stem}.txt", img_w, img_h
        )
        total_gt_boxes += len(gt_boxes)

        results = model.predict(
            source=str(img_path),
            imgsz=IMGSZ,
            conf=0.25,
            iou=0.45,
            device=device,
            verbose=False,
        )
        res = results[0]
        if hasattr(res, "speed") and "inference" in res.speed:
            inference_times_ms.append(float(res.speed["inference"]))

        if res.boxes is None or len(res.boxes) == 0 or len(gt_boxes) == 0:
            continue

        pred_boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        pred_cls = res.boxes.cls.cpu().numpy().astype(int)

        used_preds: set[int] = set()
        iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)

        for gt_idx, g_cls in enumerate(gt_cls):
            best_iou = 0.0
            best_pred_idx = -1
            for pred_idx, p_cls in enumerate(pred_cls):
                if pred_idx in used_preds or p_cls != g_cls:
                    continue
                if iou_mat[gt_idx, pred_idx] > best_iou:
                    best_iou = float(iou_mat[gt_idx, pred_idx])
                    best_pred_idx = pred_idx
            if best_pred_idx != -1 and best_iou > 0.0:
                used_preds.add(best_pred_idx)
                matched_ious.append(best_iou)

    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    mean_inf_ms = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
    return mean_iou, mean_inf_ms, len(matched_ious), total_gt_boxes


def copy_plots(train_dir: Path, test_dir: Path, results_dir: Path) -> None:
    """Eğitim ve test grafiklerini results/ klasörüne kopyalar."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Eğitim sonuç grafiği
    if (train_dir / "results.png").exists():
        shutil.copy2(train_dir / "results.png", results_dir / "results.png")

    # Test (veya yoksa train) confusion matrix ve PR eğrisi
    plot_candidates = {
        "confusion_matrix.png": [
            test_dir / "confusion_matrix.png",
            train_dir / "confusion_matrix.png",
        ],
        "PR_curve.png": [
            test_dir / "BoxPR_curve.png",
            test_dir / "PR_curve.png",
            train_dir / "BoxPR_curve.png",
            train_dir / "PR_curve.png",
        ],
    }

    for target_name, sources in plot_candidates.items():
        for src in sources:
            if src.exists():
                shutil.copy2(src, results_dir / target_name)
                break


def main() -> None:
    root = Path(__file__).resolve().parent
    data_yaml = root / "data" / "custom_coco128.yaml"
    split_summary_path = root / "data" / "split_summary.json"

    if not data_yaml.exists():
        print("[INFO] Veri seti hazırlanıyor...")
        split_summary = prepare_dataset(root)
    else:
        split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))

    device, device_desc = get_device()
    print(f"[INFO] Kullanılan cihaz: {device_desc}")

    models_dir = root / "models"
    results_dir = root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Modeli yükle ve eğit
    model = YOLO("yolov8n.pt")
    train_start = time.time()
    train_results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        optimizer=OPTIMIZER,
        lr0=LR0,
        weight_decay=WEIGHT_DECAY,
        device=device,
        workers=WORKERS,
        seed=SEED,
        project=str(root / "runs"),
        name="yolov8n_coco128",
        exist_ok=True,
    )
    train_elapsed_sec = time.time() - train_start
    train_elapsed_min = train_elapsed_sec / 60.0
    print(
        f"[INFO] Eğitim tamamlandı: {train_elapsed_sec:.2f} sn "
        f"({train_elapsed_min:.2f} dk)"
    )

    train_save_dir = Path(train_results.save_dir)
    best_pt_src = train_save_dir / "weights" / "best.pt"
    if not best_pt_src.exists():
        best_pt_src = train_save_dir / "weights" / "last.pt"

    best_pt_dst = models_dir / "best.pt"
    shutil.copy2(best_pt_src, best_pt_dst)
    print(f"[INFO] En iyi ağırlık kaydedildi: {best_pt_dst}")

    # 2. Hiç görülmemiş Test kümesi üzerinde değerlendirme
    best_model = YOLO(str(best_pt_dst))
    test_val = best_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=device,
        workers=WORKERS,
        project=str(root / "runs"),
        name="yolov8n_coco128_test",
        exist_ok=True,
    )
    test_save_dir = Path(test_val.save_dir)

    precision = float(test_val.box.mp)
    recall = float(test_val.box.mr)
    map50 = float(test_val.box.map50)
    map50_95 = float(test_val.box.map)
    f1_score = (
        (2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    # 3. Test kümesi üzerinde ortalama IoU ve Inference Time (ms/image)
    test_images_dir = root / "data" / "coco128_split" / "images" / "test"
    test_labels_dir = root / "data" / "coco128_split" / "labels" / "test"
    mean_iou, inference_ms, matched_boxes, total_gt_boxes = evaluate_test_iou_and_speed(
        best_model, test_images_dir, test_labels_dir, device
    )

    # 4. Grafikleri kopyala
    copy_plots(train_save_dir, test_save_dir, results_dir)

    # 5. Metrikleri JSON olarak kaydet
    metrics_payload = {
        "model": "YOLOv8n (yolov8n.pt)",
        "dataset": {
            "name": "COCO128",
            "total_images": split_summary.get("total_images", 128),
            "train_images": split_summary.get("train_images", 0),
            "val_images": split_summary.get("val_images", 0),
            "test_images": split_summary.get("test_images", 0),
        },
        "hyperparameters": {
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch_size": BATCH_SIZE,
            "optimizer": OPTIMIZER,
            "lr0": LR0,
            "weight_decay": WEIGHT_DECAY,
            "workers": WORKERS,
            "seed": SEED,
        },
        "environment": {
            "os": f"{platform.system()} {platform.release()}",
            "device": device_desc,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "training_time_sec": round(train_elapsed_sec, 2),
        "training_time_min": round(train_elapsed_min, 2),
        "test_metrics": {
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "mean_iou": round(mean_iou, 4),
            "matched_gt_boxes": matched_boxes,
            "total_gt_boxes": total_gt_boxes,
        },
        "inference_time_ms_per_image": round(inference_ms, 2),
    }

    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Metrikler kaydedildi: {metrics_path}")
    print(json.dumps(metrics_payload["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
