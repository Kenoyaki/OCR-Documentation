# Face & Plate Recognition System

Sistem akses kontrol otomatis untuk parkir menggunakan deteksi wajah dan plat nomor secara real-time.

> **Status:** Proof of Concept (PoC) / Pre-Production  
> **Author:** Julian Yang

---

## Deskripsi

Sistem ini melakukan autentikasi secara bersamaan dengan mendeteksi wajah pengemudi dan plat nomor kendaraan. Sistem dirancang untuk memicu PLC (Programmable Logic Controller) membuka barrier hanya ketika **kedua** faktor (wajah + plat) cocok dengan data yang diotorisasi.

---

## Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Object Detection | YOLOv11 (ONNX FP16) |
| Face Recognition | InsightFace (ArcFace) |
| OCR Plat | EasyOCR |
| Inference Engine | ONNX Runtime (GPU) |
| Bahasa | Python 3.12+ |

---

## Struktur File

```
├── inference.py          # Core runtime - deteksi & recognition
├── prepare_assets.py     # Build tool - generate database wajah & plat
├── extract_faces.py      # Helper - crop wajah dari video
├── requirements.txt      # Python dependencies
├── face_plate.yaml       # YOLO training config
├── faces/                # Folder foto referensi wajah
│   ├── NamaOrang1/
│   └── NamaOrang2/
└── augment_faces.py      # Augmentasi dataset wajah
```

---

## Cara Setup & Menjalankan

### Prerequisites
- Python 3.12+
- NVIDIA GPU dengan CUDA 11.8+ (sangat disarankan)
- Windows 10/11 atau Ubuntu 20.04+

### 1. Clone Repository

```bash
git clone https://github.com/Kenoyaki/OCR-Documentation.git
cd OCR-Documentation
```

### 2. Buat Virtual Environment

```bash
python -m venv .venv
```

### 3. Aktifkan Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

Jika berhasil, akan muncul `(.venv)` di depan prompt CMD.

### 4. Install Dependencies

```bash
cd "Face and plate recog"
pip install -r requirements.txt
```

### 5. Siapkan Data Wajah

Buat struktur folder seperti ini di dalam `faces/`:

```
faces/
├── NamaKamu/
│   ├── foto1.jpg
│   ├── foto2.jpg
│   └── foto3.jpg
└── NamaOrangLain/
    └── foto1.jpg
```

> Minimal 3-5 foto per orang, pastikan wajah jelas dan frontal.

### 6. Generate Database & Convert Model

Jalankan `prepare_assets.py` untuk membuat `faces.pkl`, `plates.pkl`, dan `face_plate_fp16.onnx`:

```bash
python prepare_assets.py --weights "yolo11n.pt" --faces_dir "faces" --plates_txt "B1234CD B5678EF"
```

> Ganti `B1234CD B5678EF` dengan plat nomor yang diotorisasi.

### 7. Jalankan Sistem

**Menggunakan webcam laptop:**

Edit bagian paling bawah `inference.py`, ubah ke:
```python
process_stream(0)
```

Lalu jalankan:
```bash
python inference.py
```

**Menggunakan IP Camera (RTSP):**
```python
process_stream("rtsp://username:password@192.168.1.x/stream?tcp")
```

> Tekan **ESC** untuk keluar dari program.

---

## Catatan Penting

- File `faces.pkl`, `plates.pkl`, dan model `.onnx` / `.pt` **tidak disertakan** di repo ini karena ukurannya besar. Generate sendiri menggunakan `prepare_assets.py`.
- Tanpa GPU, inference mungkin melebihi target latency 1 detik.
- Pastikan pencahayaan cukup — EasyOCR kurang akurat di kondisi gelap.
- InsightFace membutuhkan wajah yang relatif frontal (maks. 45 derajat).

---

## Roadmap

- [ ] Integrasi SQL Server (menggantikan `.pkl`)
- [ ] Integrasi PLC via Modbus TCP
- [ ] Dashboard monitoring multi-gate
- [ ] GUI untuk tambah wajah baru (tkinter)
- [ ] REST API untuk manajemen kredensial
