# Gerçek Dünya Fotoğrafları Değerlendirme Raporu

- **Model:** `models/best.pt`
- **Confidence Threshold:** `0.25`
- **IoU Threshold:** `0.45`
- **Toplam Fotoğraf:** `5`
- **Toplam Tespit Sayısı:** `23`
- **Ortalama Inference Süresi:** `23.69 ms/image`

| Fotoğraf | Tespit Edilen Nesneler | Güven Skorları (%) | Inference Süresi (ms) |
| :--- | :--- | :--- | :---: |
| `photo_1.jpg` | mouse, cup, potted plant, keyboard, tv, mouse, bowl, remote | mouse: %81.1, cup: %79.8, potted plant: %78.0, keyboard: %59.4, tv: %36.7, mouse: %30.1, bowl: %25.9, remote: %25.7 | 6.74 ms |
| `photo_2.jpg` | keyboard, laptop, mouse, cup, keyboard | keyboard: %85.5, laptop: %84.1, mouse: %77.8, cup: %60.0, keyboard: %42.6 | 34.87 ms |
| `photo_3.jpg` | laptop, mouse, cup, keyboard, book, chair | laptop: %91.6, mouse: %90.9, cup: %78.9, keyboard: %46.7, book: %33.6, chair: %26.8 | 40.28 ms |
| `photo_4.jpg` | cup, cup, potted plant, cup | cup: %58.2, cup: %55.2, potted plant: %49.0, cup: %28.9 | 12.01 ms |
| `photo_5.jpg` | Tespit Yok | - | 24.54 ms |
