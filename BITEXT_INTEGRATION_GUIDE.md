# Panduan Integrasi Dataset Bitext Gen AI Chatbot Customer Support

Dokumen ini menjelaskan cara mengintegrasikan dataset Bitext Gen AI Chatbot Customer Support dengan sistem chatbot SIPD yang ada.

## Tentang Dataset Bitext

Dataset Bitext Gen AI Chatbot Customer Support adalah dataset yang berisi pasangan pertanyaan/jawaban yang dihasilkan menggunakan metodologi hybrid. Dataset ini berisi data teks yang ekstensif dalam kolom 'instruction' dan 'response', dengan total 3,57 juta token. Dataset ini cocok untuk melatih model LLM untuk AI Conversational, AI Generative, dan model Question and Answering (Q&A).

Dataset ini tersedia secara gratis dan berisi lebih dari 8.000 ucapan dari 27 intent umum yang dikelompokkan menjadi 11 kategori utama. Dataset ini diformat sebagai file CSV dengan kolom-kolom berikut:

- `flags`: tag yang menjelaskan variasi bahasa
- `instruction`: permintaan pengguna dari domain Customer Service
- `category`: kategori semantik tingkat tinggi untuk intent
- `intent`: intent yang sesuai dengan instruksi pengguna
- `response`: contoh respons yang diharapkan dari asisten virtual

## Struktur Direktori

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
├── bitext_processor.py          # Processor untuk dataset Bitext
├── data_processor.py            # Processor untuk dataset SIPD
├── download_bitext_dataset.py   # Script untuk mengunduh dataset Bitext
└── integrate_bitext_dataset.py  # Script untuk mengintegrasikan dataset
```

## Langkah-langkah Integrasi

### 1. Mengunduh Dataset Bitext

Untuk mengunduh dataset Bitext, jalankan script `download_bitext_dataset.py`:

```bash
python download_bitext_dataset.py
```

Script ini akan mengunduh dataset dari Hugging Face dan menyimpannya di direktori `data/bitext/`.

### 2. Memproses Dataset Bitext

Untuk memproses dataset Bitext dan mengintegrasikannya dengan dataset SIPD, jalankan script `integrate_bitext_dataset.py`:

```bash
python integrate_bitext_dataset.py
```

Script ini akan:
1. Memproses dataset Bitext menggunakan `BitextDataProcessor`
2. Memproses dataset SIPD menggunakan `SIPDDataProcessor`
3. Menggabungkan kedua dataset menjadi satu file JSON untuk pelatihan

### 3. Menggunakan Dataset Gabungan untuk Pelatihan

Setelah integrasi selesai, Anda dapat menggunakan file `data/processed/combined_training_data.json` untuk melatih model chatbot Anda.

## Struktur Kelas BitextDataProcessor

Kelas `BitextDataProcessor` dirancang untuk memproses dataset Bitext dan mengkonversinya ke format yang kompatibel dengan sistem yang ada. Berikut adalah metode-metode utama dalam kelas ini:

- `load_csv_files()`: Memuat semua file CSV dari direktori
- `clean_text()`: Membersihkan dan menormalisasi teks
- `consolidate_dataframes()`: Menggabungkan semua dataframe dan menstandardisasi kolom
- `create_training_data()`: Mengkonversi data ke format training untuk fine-tuning
- `save_training_data()`: Menyimpan data training ke file JSON
- `process_all_data()`: Memproses semua data CSV menjadi training data

## Pemetaan Kolom

Berikut adalah pemetaan kolom dari dataset Bitext ke format standar yang digunakan oleh sistem:

| Format Standar | Kolom Bitext |
|----------------|---------------|
| ISSUE          | instruction   |
| EXPECTED       | response      |
| MENU           | category, intent |

## Catatan Penting

- Dataset Bitext tidak memiliki kolom yang setara dengan `NOTE_BY_DEV` dan `NOTE_BY_QA` yang ada di dataset SIPD.
- Kolom `category` dan `intent` dari dataset Bitext digabungkan untuk membentuk kolom `MENU` dalam format standar.
- Format output JSON mengikuti format yang digunakan untuk fine-tuning model, dengan peran "user" dan "assistant".

## Troubleshooting

### Dataset Tidak Dapat Diunduh

Jika dataset tidak dapat diunduh secara otomatis, Anda dapat mengunduhnya secara manual dari Hugging Face dan menempatkannya di direktori `data/bitext/`.

### Error Saat Memproses Dataset

Jika terjadi error saat memproses dataset, periksa log di `logs/bitext_integration.log` untuk informasi lebih lanjut.

## Referensi

- [Bitext Gen AI Chatbot Customer Support Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- [GitHub Repository Bitext](https://github.com/bitext/customer-support-llm-chatbot-training-dataset)