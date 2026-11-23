"""
COCO128 veri setini indirir ve deterministic (seed=42) olarak
%70 Train, %15 Validation, %15 Test kümelerine ayırır.
Ayrıca data/custom_coco128.yaml dosyasını otomatik oluşturur.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import yaml
from sklearn.model_selection import train_test_split

SEED = 42
COCO128_URL = "https://ultralytics.com/assets/coco128.zip"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}


def download_and_extract_coco128(data_dir: Path) -> Path:
    """COCO128 arşivini indirir ve data_dir/coco128 altına çıkartır."""
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "coco128"
    zip_path = data_dir / "coco128.zip"

    if (raw_dir / "images" / "train2017").exists():
        print(f"[INFO] COCO128 zaten mevcut: {raw_dir}")
        return raw_dir

    if not zip_path.exists():
        print(f"[INFO] COCO128 indiriliyor: {COCO128_URL}")
        urlretrieve(COCO128_URL, zip_path)
        print(f"[INFO] İndirme tamamlandı: {zip_path}")

    print(f"[INFO] Arşiv çıkartılıyor: {data_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    if not (raw_dir / "images" / "train2017").exists():
        raise FileNotFoundError(
            f"Beklenen görüntü klasörü bulunamadı: {raw_dir / 'images' / 'train2017'}"
        )
    return raw_dir


def copy_split_files(
    image_paths: list[Path],
    src_labels_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
) -> int:
    """Görüntüleri ve varsa karşılık gelen YOLO etiket dosyalarını kopyalar."""
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    missing_labels = 0
    for img_path in image_paths:
        shutil.copy2(img_path, dst_images_dir / img_path.name)
        label_src = src_labels_dir / f"{img_path.stem}.txt"
        label_dst = dst_labels_dir / f"{img_path.stem}.txt"
        if label_src.exists():
            shutil.copy2(label_src, label_dst)
        else:
            # Etiketsiz (background) görüntüler için boş etiket dosyası oluştur
            label_dst.write_text("", encoding="utf-8")
            missing_labels += 1
    return missing_labels


def prepare_dataset(project_root: Path | None = None) -> dict[str, int]:
    """Veri setini %70 Train, %15 Val, %15 Test olarak böler ve YAML üretir."""
    root = project_root or Path(__file__).resolve().parent
    data_dir = root / "data"
    raw_dir = download_and_extract_coco128(data_dir)

    src_images_dir = raw_dir / "images" / "train2017"
    src_labels_dir = raw_dir / "labels" / "train2017"

    all_images = sorted(
        p for p in src_images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not all_images:
        raise RuntimeError(f"Görüntü bulunamadı: {src_images_dir}")

    train_imgs, temp_imgs = train_test_split(
        all_images, test_size=0.30, random_state=SEED, shuffle=True
    )
    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=0.50, random_state=SEED, shuffle=True
    )

    split_dir = data_dir / "coco128_split"
    if split_dir.exists():
        shutil.rmtree(split_dir)

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs,
    }

    for split_name, imgs in splits.items():
        missing = copy_split_files(
            imgs,
            src_labels_dir,
            split_dir / "images" / split_name,
            split_dir / "labels" / split_name,
        )
        print(
            f"[INFO] {split_name.upper():5s}: {len(imgs)} görüntü "
            f"(arka plan/etiketsiz: {missing})"
        )

    yaml_path = data_dir / "custom_coco128.yaml"
    yaml_content = {
        "path": split_dir.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(COCO_NAMES),
        "names": COCO_NAMES,
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f, sort_keys=False, allow_unicode=True)
    print(f"[INFO] YAML oluşturuldu: {yaml_path}")

    summary = {
        "total_images": len(all_images),
        "train_images": len(train_imgs),
        "val_images": len(val_imgs),
        "test_images": len(test_imgs),
        "seed": SEED,
    }
    summary_path = data_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    prepare_dataset()
