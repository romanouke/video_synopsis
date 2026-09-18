# Video Synopsis Demo

Implementasi demonstrasi konsep **Video Synopsis** — memadatkan kejadian dari video panjang dengan menyusun ulang **object tubes** pada timeline sehingga beberapa kejadian dari waktu berbeda dapat tampil bersamaan, selama collision tidak signifikan.

**Fokus**: Rearrangement temporal dari moving objects, bukan keyframe summarization.

---

## Fitur Utama

- **Upload video** melalui web UI
- **Motion Detection + BoT-SORT Tracker**: Deteksi gerakan dan tracking ID stabil per object
- **Object Tube Extraction**: Kelompokkan deteksi per frame sebagai "tubes" (trajectory per object)
- **Temporal Rearrangement**: Greedy scheduling untuk menumpuk tubes tanpa collision signifikan
- **Background Generation**: Average running dari frames untuk latar belakang statis
- **Foreground Extraction & Compositing**: Extract objek bergerak, composite ke background
- **Video Encoding**: Output MP4 yang sudah dikompresi temporal
- **Web UI**: Upload → Processing → Preview + Download hasil


---

## Setup

### 1. Clone atau download project

```bash
cd c:\laragon\www\video_synopsis
```

### 2. Install Python 3.11+

Gunakan Python 3.11 atau lebih baru. Project ini dikembangkan dan ditest dengan Python 3.11.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies utama:
- `fastapi==0.115.6` — Web framework
- `uvicorn[standard]==0.34.0` — ASGI server
- `opencv-python==4.10.0.84` — Video processing & motion detection
- `numpy==2.2.1` — Numerical computing
- `jinja2==3.1.5` — Template rendering
- `python-multipart==0.0.20` — File upload support

### 4. (Opsional) Install YOLO

Jika ingin menggunakan YOLO v8 untuk deteksi object yang lebih akurat:

```bash
pip install ultralytics
```

---

## Menjalankan Project

### Start Server

```bash
python -m uvicorn app:app --reload
```

Atau gunakan Python 3.11 secara eksplisit:

```bash
C:\Users\{username}\AppData\Local\Programs\Python\Python311\python.exe -m uvicorn app:app --reload
```

Server akan berjalan di `http://127.0.0.1:8000`.

### Akses Web UI

Buka browser dan kunjungi:

```
http://127.0.0.1:8000
```

---

## Cara Menggunakan

### 1. Upload Video

1. Di halaman utama, klik **"Pilih video input"** atau drag-drop file video
2. Format yang didukung: MP4, MOV, AVI, dan format lainnya yang didukung OpenCV
3. Klik **"Proses Video Synopsis"**

### 2. Tunggu Proses Selesai

- Server akan memproses video:
  - Deteksi gerakan / YOLO detection
  - Tracking per frame
  - Filtering tube (minimal 4 frame stable)
  - Scheduling temporal (rearrangement greedy)
  - Compositing & encoding MP4

### 3. Download Hasil

Setelah selesai, halaman akan menampilkan:
- **Preview video** (synopsis MP4 yang sudah dikompresi temporal)
- **Background image** (running average frame)
- **Statistik**: jumlah tube, frame rate, compression ratio, runtime

Klik **"Download output MP4"** atau **"Download background.jpg"** untuk menyimpan file.

---

## Konfigurasi

### Motion Detection (Default)

Default menggunakan motion detection berbasis background subtraction (MOG2). Cocok untuk video dengan background statis.

### YOLO Detection (Opsional)

Edit `app.py` atau `synopsis/pipeline.py` untuk mengaktifkan YOLO:

```python
# Di app.py, saat membuat processor:
config = SynopsisConfig(
    use_yolo=True,
    yolo_model="yolov8n.pt",  # nano model for faster inference
    yolo_conf_threshold=0.3,
)
processor = VideoSynopsisProcessor(config)
```

Model YOLO yang tersedia:
- `yolov8n.pt` — Nano (tercepat, CPU-friendly)
- `yolov8s.pt` — Small
- `yolov8m.pt` — Medium
- `yolov8l.pt` — Large

Model akan otomatis didownload saat pertama kali digunakan.

### BoT-SORT Tracker Config

Modifikasi tracking behavior di `SynopsisConfig`:

```python
config = SynopsisConfig(
    bot_max_lost=18,  # frames sebelum track dihapus
    bot_min_hits=2,   # hits sebelum track dianggap confirmed
    bot_high_confidence_threshold=0.45,
    bot_low_confidence_threshold=0.15,
    bot_iou_threshold=0.15,  # IoU minimum untuk asosiasi
    bot_appearance_threshold=0.25,  # appearance similarity minimum
    bot_cost_threshold=0.95,  # cost maksimum untuk matching
)
```

### Temporal Rearrangement Config

```python
config = SynopsisConfig(
    collision_iou_threshold=0.08,  # IoU threshold untuk deteksi collision
    min_tube_length=4,  # minimum frame per tube untuk dipertahankan
)
```

---

## Struktur Project

```
video_synopsis/
├── app.py                    # FastAPI application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── synopsis/                 # Core pipeline package
│   ├── __init__.py
│   ├── types.py             # Data structures (Detection, Track, etc.)
│   ├── utils.py             # Utility functions (bbox ops, resizing, etc.)
│   ├── detection.py         # MotionDetector & YoloDetector
│   ├── tracking.py          # BoT-SORT tracker
│   ├── pipeline.py          # VideoSynopsisProcessor (main pipeline)
│   └── worker.py            # (Optional) Background job manager
│
├── templates/               # Jinja2 templates
│   ├── base.html           # Base template
│   ├── index.html          # Upload page
│   ├── result.html         # Result page
│   └── job.html            # (Optional) Job status page
│
├── static/                  # CSS & static assets
│   └── styles.css
│
├── uploads/                 # Temporary upload folder (git-ignored)
├── outputs/                 # Output folder (git-ignored)
└── .vscode/                 # VS Code settings
    └── settings.json
```

---

## Pipeline Architecture

```
INPUT VIDEO
    ↓
┌─────────────────────────────────────┐
│ VIDEO PREPROCESSING                 │
│ - Resize to max_width (640px)      │
│ - Frame sampling (stride)           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ OBJECT DETECTION (per frame)        │
│ - Motion Detection (MOG2)           │
│ - OR YOLO v8 (if enabled)          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ OBJECT TRACKING (across frames)     │
│ - BoT-SORT tracker                  │
│ - Kalman prediction + appearance    │
│ - Two-stage association             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ TUBE EXTRACTION & FILTERING         │
│ - Group observations per track ID   │
│ - Filter by min_tube_length         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ TEMPORAL REARRANGEMENT              │
│ - Greedy scheduling algorithm       │
│ - Check collision_iou_threshold     │
│ - Stack tubes on new timeline       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ BACKGROUND GENERATION               │
│ - Running average of all frames     │
│ - Saved as JPG                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ COMPOSITING & ENCODING              │
│ - For each output frame:            │
│   1. Start with background          │
│   2. Alpha-blend active tubes       │
│   3. Write frame to MP4             │
└─────────────────────────────────────┘
    ↓
OUTPUT VIDEO SYNOPSIS (compressed temporal)
    ↓
PREVIEW + DOWNLOAD
```

---

## Contoh Testing

### 1. Buat Video Test Dummy

```powershell
# Buat video 10 detik dengan lingkaran hijau bergerak
python -c "
import cv2
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_video.mp4', fourcc, 15.0, (640, 480))

for i in range(150):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cx = 100 + (i % 200)
    cy = 240
    cv2.circle(frame, (cx, cy), 30, (0, 255, 0), -1)
    out.write(frame)

out.release()
print('test_video.mp4 created')
"
```

### 2. Upload dan Process

1. Buka http://127.0.0.1:8000
2. Upload `test_video.mp4`
3. Tunggu proses selesai (biasanya 30-60 detik untuk video 10 detik)

### 3. Download Hasil

Hasil akan tersimpan di:
- `outputs/{job_id}/synopsis.mp4` — Video hasil
- `outputs/{job_id}/background.jpg` — Background
- `uploads/{job_id}/test_video.mp4` — File upload asli

---

## Performance Notes

### Processing Time

- **Per frame**: ~100-300ms (motion detection) atau ~500-1000ms (YOLO)
- **Typical 1-minute video**: 2-5 menit processing time (motion), 5-10 menit (YOLO)
- **Bottlenecks**: 
  - YOLO inference (jika diaktifkan)
  - Kalman tracking update
  - Temporal scheduling (quadratic worst-case)

### Memory Usage

- **Per video**: ~500MB-1GB (depends on resolution)
- **Optimisasi**: Video diproses frame-by-frame, bukan loaded seluruhnya

### Rekomendasi Hardware

- **CPU**: Minimal Intel i5 / Ryzen 5
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB free untuk temp + output

---

## Troubleshooting

### Server tidak start

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

```bash
# Pastikan dependencies terpasang:
pip install -r requirements.txt

# Atau gunakan Python 3.11 secara eksplisit:
C:\Users\{username}\AppData\Local\Programs\Python\Python311\python.exe -m pip install -r requirements.txt
```

### Video tidak terdeteksi

**Masalah**: Tidak ada tube yang terdeteksi, pesan error "Tidak ada tube yang cukup stabil"

**Solusi**:
- Pastikan video punya gerakan yang jelas (background statis, objek bergerak)
- Coba turunkan `min_area` di `MotionDetectorConfig`
- Atau gunakan YOLO detector untuk deteksi yang lebih kuat

### Processing time lama

**Masalah**: Processing lambat

**Solusi**:
- Gunakan motion detection (default) bukan YOLO
- Turunkan `max_width` untuk resize frame lebih kecil
- Naikkan `target_fps` (frame sampling) untuk skip frame lebih banyak

### YOLO model error

**Error**: `RuntimeError: CUDA out of memory` atau model tidak terdownload

**Solusi**:
```bash
# Download manual:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Atau gunakan model nano yang lebih ringan:
config = SynopsisConfig(yolo_model="yolov8n.pt", use_yolo=True)
```

---

## API Endpoints

### `GET /`

Halaman utama (upload form).

### `POST /process`

Upload & process video. Return HTML halaman hasil atau error message.

**Multipart form data**:
- `file`: Video file

**Response**: HTML halaman dengan preview, atau error page

---

## Lisensi & Attribution

Project ini adalah demonstrasi edukasi konsep Video Synopsis:

- **Motion Detection**: OpenCV MOG2
- **YOLO**: Ultralytics (optional)
- **Tracking**: BoT-SORT inspired architecture
- **Temporal Optimization**: Custom greedy scheduling
