# Integrasi Dataset Bitext Gen AI Chatbot Customer Support

Proyek ini menyediakan alat dan script untuk mengintegrasikan dataset Bitext Gen AI Chatbot Customer Support dengan sistem chatbot SIPD yang ada.

## Daftar Isi

- [Pendahuluan](#pendahuluan)
- [Struktur Proyek](#struktur-proyek)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Fitur](#fitur)
- [Dokumentasi](#dokumentasi)
- [Troubleshooting](#troubleshooting)

## Pendahuluan

Dataset Bitext Gen AI Chatbot Customer Support adalah dataset yang berisi pasangan pertanyaan/jawaban yang dihasilkan menggunakan metodologi hybrid. Dataset ini berisi data teks yang ekstensif dalam kolom 'instruction' dan 'response', dengan total 3,57 juta token. Dataset ini cocok untuk melatih model LLM untuk AI Conversational, AI Generative, dan model Question and Answering (Q&A).

Proyek ini menyediakan alat untuk:
1. Mengunduh dataset Bitext
2. Memproses dataset Bitext ke format yang kompatibel dengan sistem yang ada
3. Mengintegrasikan dataset Bitext dengan dataset SIPD yang ada
4. Menguji integrasi dengan sistem RAG
5. Membandingkan statistik antara dataset SIPD dan Bitext

## Struktur Proyek

```
ChatbotAI/
├── data/
│   ├── bitext/                  # Direktori untuk dataset Bitext
│   │   └── bitext-customer-support-dataset.csv
│   ├── csv/                     # Direktori untuk dataset SIPD
│   │   └── sample_sipd_issues.csv
│   └── processed/               # Direktori untuk data yang telah diproses
│       ├── bitext_training_data.json
│       ├── sipd_training_data.json
│       └── combined_training_data.json
├── logs/                        # Direktori untuk file log
├── reports/                     # Direktori untuk laporan perbandingan dataset
├── bitext_processor.py          # Processor untuk dataset Bitext
├── data_processor.py            # Processor untuk dataset SIPD
├── download_bitext_dataset.py   # Script untuk mengunduh dataset Bitext
├── integrate_bitext_dataset.py  # Script untuk mengintegrasikan dataset
├── test_bitext_rag_integration.py # Script untuk menguji integrasi dengan RAG
├── compare_datasets.py          # Script untuk membandingkan dataset
├── BITEXT_INTEGRATION_GUIDE.md  # Panduan integrasi dataset Bitext
└── README_BITEXT_INTEGRATION.md # File README ini
```

## Instalasi

### Prasyarat

- Python 3.8 atau lebih tinggi
- pip (Python package installer)

### Dependensi

Instal dependensi yang diperlukan dengan menjalankan:

```bash
pip install pandas loguru requests matplotlib numpy
```

Jika Anda ingin menggunakan fitur RAG, Anda juga perlu menginstal:

```bash
pip install chromadb sentence-transformers
```

## Penggunaan

### 1. Mengunduh Dataset Bitext

Untuk mengunduh dataset Bitext, jalankan:

```bash
python download_bitext_dataset.py
```

Script ini akan mengunduh dataset dari Hugging Face dan menyimpannya di direktori `data/bitext/`.

### 2. Memproses dan Mengintegrasikan Dataset

Untuk memproses dataset Bitext dan mengintegrasikannya dengan dataset SIPD, jalankan:

```bash
python integrate_bitext_dataset.py
```

Script ini akan memproses kedua dataset dan menggabungkannya menjadi satu file JSON untuk pelatihan.

### 3. Menguji Integrasi dengan RAG

Untuk menguji integrasi dataset Bitext dengan sistem RAG, jalankan:

```bash
python test_bitext_rag_integration.py
```

Script ini akan menguji sistem RAG dengan beberapa query sampel menggunakan dataset Bitext.

### 4. Membandingkan Dataset

Untuk membandingkan statistik antara dataset SIPD dan Bitext, jalankan:

```bash
python compare_datasets.py
```

Script ini akan menghasilkan laporan perbandingan dan grafik di direktori `reports/`.

## Fitur

### BitextDataProcessor

Kelas `BitextDataProcessor` dirancang untuk memproses dataset Bitext dan mengkonversinya ke format yang kompatibel dengan sistem yang ada. Fitur utama:

- Memuat file CSV dari direktori
- Membersihkan dan menormalisasi teks
- Menggabungkan dataframe dan menstandardisasi kolom
- Mengkonversi data ke format training untuk fine-tuning
- Menyimpan data training ke file JSON

### Integrasi dengan RAG

Script `test_bitext_rag_integration.py` menunjukkan cara mengintegrasikan dataset Bitext dengan sistem RAG yang ada. Fitur utama:

- Mengkonversi data training ke format dokumen RAG
- Menambahkan dokumen ke sistem RAG
- Menguji sistem RAG dengan query sampel

### Perbandingan Dataset

Script `compare_datasets.py` membandingkan statistik antara dataset SIPD dan Bitext. Fitur utama:

- Menghitung statistik teks (panjang rata-rata, jumlah kata, dll.)
- Mengekstrak kategori menu dari pesan pengguna
- Menghasilkan laporan perbandingan dalam format JSON
- Membuat grafik perbandingan

## Dokumentasi

Untuk informasi lebih lanjut tentang integrasi dataset Bitext, lihat [BITEXT_INTEGRATION_GUIDE.md](BITEXT_INTEGRATION_GUIDE.md).

## Troubleshooting

### Dataset Tidak Dapat Diunduh

Jika dataset tidak dapat diunduh secara otomatis, Anda dapat mengunduhnya secara manual dari [Hugging Face](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) dan menempatkannya di direktori `data/bitext/`.

### Error Saat Memproses Dataset

Jika terjadi error saat memproses dataset, periksa file log di direktori `logs/` untuk informasi lebih lanjut.

### Masalah dengan Integrasi RAG

Jika Anda mengalami masalah dengan integrasi RAG, pastikan Anda telah menginstal semua dependensi yang diperlukan dan dataset telah diproses dengan benar.