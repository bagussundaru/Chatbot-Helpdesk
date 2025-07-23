#!/usr/bin/env python3
"""
Setup script untuk SIPD AI Chatbot
Script ini membantu setup environment dan deployment
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, Any

def create_directories():
    """Buat direktori yang diperlukan"""
    directories = [
        "data/csv",
        "data/processed", 
        "vector_store",
        "logs",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False
    return True

def setup_environment():
    """Setup environment variables"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✓ .env file already exists")
        return True
        
    # Copy from example
    example_file = Path(".env.example")
    if example_file.exists():
        with open(example_file, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✓ Created .env file from .env.example")
        print("⚠️  Please update .env file with your actual API keys and configuration")
        return True
    else:
        print("✗ .env.example file not found")
        return False

def create_sample_csv_data():
    """Buat sample data CSV untuk testing"""
    sample_data = [
        {
            "MENU": "Login/Akses",
            "ISSUE": "Tidak bisa login ke SIPD, muncul error 500",
            "EXPECTED": "User dapat login dengan normal",
            "NOTE BY DEV": "Cek koneksi database dan clear browser cache",
            "NOTE BY QA": "Pastikan username dan password benar, coba browser lain"
        },
        {
            "MENU": "Penganggaran",
            "ISSUE": "Error saat input DPA, data tidak tersimpan",
            "EXPECTED": "DPA dapat diinput dan tersimpan dengan benar",
            "NOTE BY DEV": "Validasi format data DPA dan cek koneksi",
            "NOTE BY QA": "Pastikan semua field mandatory terisi"
        },
        {
            "MENU": "Pelaporan",
            "ISSUE": "Laporan tidak bisa di-export ke Excel",
            "EXPECTED": "Laporan dapat di-export ke format Excel",
            "NOTE BY DEV": "Update library export dan cek permission file",
            "NOTE BY QA": "Coba export dengan data yang lebih sedikit"
        },
        {
            "MENU": "Penatausahaan",
            "ISSUE": "SPP tidak bisa dibuat, tombol simpan tidak aktif",
            "EXPECTED": "SPP dapat dibuat dan disimpan",
            "NOTE BY DEV": "Cek validasi form dan JavaScript error",
            "NOTE BY QA": "Pastikan semua data pendukung sudah lengkap"
        },
        {
            "MENU": "Akuntansi",
            "ISSUE": "Jurnal otomatis tidak terbentuk setelah posting",
            "EXPECTED": "Jurnal otomatis terbentuk sesuai transaksi",
            "NOTE BY DEV": "Cek konfigurasi akun dan mapping jurnal",
            "NOTE BY QA": "Verifikasi setup chart of account"
        }
    ]
    
    import pandas as pd
    
    csv_file = Path("data/csv/sample_sipd_issues.csv")
    df = pd.DataFrame(sample_data)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    print(f"✓ Created sample CSV data: {csv_file}")
    return True

def test_local_setup():
    """Test local setup"""
    print("\nTesting local setup...")
    
    try:
        # Test imports
        from config import settings
        from data_processor import SIPDDataProcessor
        from rag_system import SIPDRAGSystem
        from nebius_client import NebiusAIClient
        from chatbot_engine import SIPDChatbotEngine
        
        print("✓ All modules imported successfully")
        
        # Test data processor
        processor = SIPDDataProcessor("data/csv")
        training_data = processor.process_all_data()
        print(f"✓ Data processor: {len(training_data)} training examples")
        
        # Test RAG system (without Nebius dependency)
        rag = SIPDRAGSystem()
        stats = rag.get_collection_stats()
        print(f"✓ RAG system initialized: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Setup test failed: {e}")
        return False

def setup_modal_deployment():
    """Setup Modal.com deployment"""
    print("\nSetting up Modal.com deployment...")
    
    try:
        # Check if modal is installed
        subprocess.run(["modal", "--version"], check=True, capture_output=True)
        print("✓ Modal CLI is installed")
        
        # Instructions for Modal setup
        print("\nModal.com deployment setup:")
        print("1. Install Modal CLI: pip install modal")
        print("2. Login to Modal: modal token new")
        print("3. Create secrets in Modal dashboard:")
        print("   - Secret name: sipd-chatbot-secrets")
        print("   - Add environment variables from .env file")
        print("4. Deploy: modal deploy modal_deployment.py")
        print("5. Run: modal run modal_deployment.py")
        
        return True
        
    except subprocess.CalledProcessError:
        print("⚠️  Modal CLI not found. Install with: pip install modal")
        return False
    except Exception as e:
        print(f"✗ Modal setup error: {e}")
        return False

def create_readme():
    """Buat file README dengan instruksi lengkap"""
    readme_content = """
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
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✓ Created README.md")

def main():
    """Main setup function"""
    print("🚀 SIPD AI Chatbot Setup")
    print("=" * 50)
    
    steps = [
        ("Creating directories", create_directories),
        ("Installing dependencies", install_dependencies),
        ("Setting up environment", setup_environment),
        ("Creating sample data", create_sample_csv_data),
        ("Testing setup", test_local_setup),
        ("Setting up Modal deployment", setup_modal_deployment),
        ("Creating README", create_readme)
    ]
    
    success_count = 0
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        try:
            if step_func():
                success_count += 1
            else:
                print(f"⚠️  {step_name} completed with warnings")
        except Exception as e:
            print(f"✗ {step_name} failed: {e}")
    
    print(f"\n🎉 Setup completed: {success_count}/{len(steps)} steps successful")
    
    if success_count >= len(steps) - 1:  # Allow 1 failure
        print("\n✅ Setup berhasil! Langkah selanjutnya:")
        print("1. Edit file .env dengan API keys Anda")
        print("2. Letakkan file CSV aduan di folder data/csv/")
        print("3. Jalankan: python app.py")
        print("4. Akses: http://localhost:8000")
    else:
        print("\n❌ Setup tidak lengkap. Periksa error di atas.")

if __name__ == "__main__":
    main()