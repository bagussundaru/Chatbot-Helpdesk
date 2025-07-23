
# SIPD AI Chatbot - Prototipe Help Desk Cerdas

## Deskripsi
Chatbot AI cerdas dan manusiawi untuk help desk SIPD (Sistem Informasi Pemerintahan Daerah) yang menggunakan teknologi RAG (Retrieval-Augmented Generation) dan fine-tuned model dari Nebius AI Studio.

## Fitur Utama
- 🤖 Respons cerdas dan manusiawi
- 📚 Knowledge base dari data historis aduan
- 🔍 RAG system untuk pencarian konteks relevan
- 🎯 Klasifikasi intent dan analisis sentimen
- 📊 Dashboard monitoring dan statistik
- 🚀 Deployment di Modal.com dengan autoscaling
- 💬 Interface web yang responsif

## Arsitektur
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI        │    │   Nebius AI     │
│   (Web UI)      │◄──►│   Backend        │◄──►│   Studio        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   RAG System     │
                       │   (ChromaDB)     │
                       └──────────────────┘
```

## Setup dan Instalasi

### 1. Persiapan Environment
```bash
# Clone atau download project
cd ChatbotAI

# Jalankan setup otomatis
python setup.py
```

### 2. Konfigurasi
1. Edit file `.env` dengan API keys Anda:
   ```
   NEBIUS_API_KEY=your_nebius_api_key
   NEBIUS_MODEL_ID=your_fine_tuned_model_id
   ```

2. Letakkan file CSV aduan SIPD di folder `data/csv/`

### 3. Jalankan Lokal
```bash
# Jalankan server development
python app.py

# Atau dengan uvicorn
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Akses aplikasi di: http://localhost:8000

### 4. Deployment ke Modal.com
```bash
# Install Modal CLI
pip install modal

# Login ke Modal
modal token new

# Setup secrets di Modal dashboard
# Nama secret: sipd-chatbot-secrets

# Deploy aplikasi
modal deploy modal_deployment.py
```

## Struktur Project
```
ChatbotAI/
├── app.py                 # FastAPI application
├── chatbot_engine.py      # Core chatbot engine
├── nebius_client.py       # Nebius AI integration
├── rag_system.py          # RAG implementation
├── data_processor.py      # CSV data processing
├── config.py              # Configuration settings
├── modal_deployment.py    # Modal.com deployment
├── setup.py               # Setup script
├── requirements.txt       # Dependencies
├── .env.example          # Environment template
├── data/
│   └── csv/              # CSV files aduan SIPD
├── vector_store/         # Vector database
└── logs/                 # Application logs
```

## API Endpoints

### Chat
- `POST /chat` - Send message to chatbot
- `GET /chat/history/{session_id}` - Get conversation history
- `DELETE /chat/history/{session_id}` - Clear conversation history

### Feedback & Support
- `POST /feedback` - Submit user feedback
- `POST /escalate` - Escalate to human agent

### System
- `GET /health` - Health check
- `GET /stats` - System statistics

## Penggunaan

### 1. Chat Interface
Buka browser dan akses aplikasi. Interface chat akan memungkinkan Anda:
- Mengirim pesan ke chatbot
- Melihat respons dengan konteks dari knowledge base
- Mendapat saran tindakan lanjutan
- Memberikan feedback

### 2. API Integration
```python
import requests

# Send chat message
response = requests.post('http://localhost:8000/chat', json={
    'message': 'Saya tidak bisa login ke SIPD',
    'session_id': 'user_123'
})

result = response.json()
print(result['response'])
```

### 3. Data Processing
```python
from data_processor import SIPDDataProcessor

# Process CSV files
processor = SIPDDataProcessor('data/csv')
training_data = processor.process_all_data()
```

## Kustomisasi

### 1. Menambah Data Training
1. Letakkan file CSV baru di `data/csv/`
2. Pastikan kolom sesuai: MENU, ISSUE, EXPECTED, NOTE BY DEV, NOTE BY QA
3. Restart aplikasi untuk memproses data baru

### 2. Fine-tuning Model
1. Export training data: `python data_processor.py`
2. Upload ke Nebius AI Studio
3. Lakukan fine-tuning dengan data JSON
4. Update `NEBIUS_MODEL_ID` di `.env`

### 3. Kustomisasi Persona
Edit system prompt di `nebius_client.py` untuk mengubah karakteristik chatbot.

## Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### System Statistics
```bash
curl http://localhost:8000/stats
```

## Troubleshooting

### Common Issues
1. **Import Error**: Pastikan semua dependencies terinstall
2. **API Error**: Cek API key Nebius di file `.env`
3. **Vector DB Error**: Hapus folder `vector_store` dan restart
4. **CSV Processing Error**: Cek format dan encoding file CSV

### Logs
Cek file log di folder `logs/` untuk debugging detail.

## Kontribusi
1. Fork repository
2. Buat feature branch
3. Commit changes
4. Push ke branch
5. Create Pull Request

## Lisensi
MIT License - Lihat file LICENSE untuk detail.

## Support
Untuk bantuan teknis, buat issue di repository atau hubungi tim development.
