# Panduan Deployment SIPD AI Chatbot

## Overview
Panduan lengkap untuk deployment SIPD AI Chatbot dari development hingga production.

## Arsitektur Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                    Production Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐ │
│  │   Users     │    │   Modal.com  │    │   Nebius AI     │ │
│  │  (Browser)  │◄──►│   FastAPI    │◄──►│   Studio        │ │
│  └─────────────┘    │   Backend    │    │   (Fine-tuned)  │ │
│                     └──────────────┘    └─────────────────┘ │
│                            │                                │
│                            ▼                                │
│                   ┌──────────────┐                         │
│                   │   ChromaDB   │                         │
│                   │ Vector Store │                         │
│                   └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Tahap 1: Development Setup

### 1.1 Local Development
```bash
# Clone project
git clone <repository-url>
cd ChatbotAI

# Setup environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

# Install dependencies
pip install fastapi uvicorn pydantic pydantic-settings
pip install loguru python-dotenv pandas

# Run demo version
python simple_app.py
```

### 1.2 Data Preparation
```bash
# Letakkan file CSV di data/csv/
# Format yang diperlukan:
# - MENU: Kategori modul SIPD
# - ISSUE: Deskripsi masalah
# - EXPECTED: Solusi yang diharapkan
# - NOTE BY DEV: Catatan developer
# - NOTE BY QA: Catatan QA

# Process data
python data_processor.py
```

## Tahap 2: AI Model Setup (Nebius AI Studio)

### 2.1 Account Setup
1. Daftar di https://studio.nebius.com/
2. Buat project baru
3. Generate API key
4. Setup billing (jika diperlukan)

### 2.2 Fine-tuning Process
```bash
# 1. Prepare training data
python data_processor.py

# 2. Upload sipd_training_data.json ke Nebius Studio
# 3. Pilih base model (recommended: Mistral-7B-Instruct)
# 4. Configure hyperparameters:
#    - learning_rate: 2e-5
#    - batch_size: 4
#    - num_epochs: 3
#    - max_length: 2048

# 5. Start fine-tuning job
# 6. Monitor training progress
# 7. Deploy model to inference endpoint
```

### 2.3 Model Configuration
```env
# Update .env file
NEBIUS_API_KEY=your_api_key_here
NEBIUS_BASE_URL=https://api.studio.nebius.ai/v1
NEBIUS_MODEL_ID=your_fine_tuned_model_id
```

## Tahap 3: Production Deployment (Modal.com)

### 3.1 Modal.com Setup
```bash
# Install Modal CLI
pip install modal

# Login to Modal
modal token new

# Verify installation
modal --version
```

### 3.2 Secrets Configuration
1. Buka Modal dashboard: https://modal.com/
2. Buat secret baru: `sipd-chatbot-secrets`
3. Tambahkan environment variables:
   ```
   NEBIUS_API_KEY=your_nebius_api_key
   NEBIUS_BASE_URL=https://api.studio.nebius.ai/v1
   NEBIUS_MODEL_ID=your_fine_tuned_model_id
   DATABASE_URL=postgresql://...
   ```

### 3.3 Deployment Process
```bash
# Deploy to Modal
modal deploy modal_deployment.py

# Initialize knowledge base (run once)
modal run modal_deployment.py::initialize_knowledge_base

# Check deployment status
modal app list
```

### 3.4 Production URLs
Setelah deployment berhasil, Modal akan memberikan URLs:
```
https://your-app-name--root.modal.run/          # Chat interface
https://your-app-name--chat.modal.run/          # Chat API
https://your-app-name--health.modal.run/        # Health check
```

## Tahap 4: Monitoring & Maintenance

### 4.1 Health Monitoring
```bash
# Check application health
curl https://your-app-name--health.modal.run/

# Monitor logs
modal logs your-app-name

# Check resource usage
modal stats your-app-name
```

### 4.2 Performance Monitoring
```python
# monitoring_script.py
import requests
import time

def monitor_chatbot():
    endpoint = "https://your-app-name--health.modal.run/"
    
    while True:
        try:
            response = requests.get(endpoint, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data['status']}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    monitor_chatbot()
```

### 4.3 Log Analysis
```bash
# View recent logs
modal logs your-app-name --follow

# Filter error logs
modal logs your-app-name | grep ERROR

# Export logs for analysis
modal logs your-app-name > chatbot_logs.txt
```

## Tahap 5: Scaling & Optimization

### 5.1 Auto-scaling Configuration
```python
# Update modal_deployment.py
@app.function(
    image=image,
    secrets=secrets,
    cpu=2.0,
    memory=4096,
    timeout=60,
    container_idle_timeout=300,
    allow_concurrent_inputs=50,  # Increase for higher load
    max_containers=10,           # Set max containers
    min_containers=1             # Keep warm containers
)
```

### 5.2 Performance Optimization
```python
# Caching strategy
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_response_generation(query_hash):
    # Cache frequent responses
    pass

# Connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

### 5.3 Cost Optimization
```yaml
# Resource optimization
CPU: 1.0-2.0 (adjust based on load)
Memory: 2048-4096 MB
Timeout: 30-60 seconds
Idle Timeout: 300 seconds
Concurrency: 10-50 requests
```

## Tahap 6: Security & Compliance

### 6.1 Security Best Practices
```python
# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, message: ChatMessage):
    # Chat logic
    pass

# Input validation
from pydantic import validator

class ChatMessage(BaseModel):
    message: str
    
    @validator('message')
    def validate_message(cls, v):
        if len(v) > 1000:
            raise ValueError('Message too long')
        return v
```

### 6.2 Data Privacy
```python
# Data anonymization
def anonymize_session_data(session_data):
    # Remove PII from logs
    anonymized = session_data.copy()
    anonymized['user_id'] = hash(session_data['user_id'])
    return anonymized

# Secure storage
from cryptography.fernet import Fernet

key = Fernet.generate_key()
cipher = Fernet(key)

def encrypt_sensitive_data(data):
    return cipher.encrypt(data.encode())
```

## Tahap 7: Backup & Recovery

### 7.1 Data Backup
```bash
# Vector database backup
modal run backup_script.py

# Conversation history backup
modal run export_conversations.py

# Model artifacts backup
cp -r vector_store/ backup/vector_store_$(date +%Y%m%d)/
```

### 7.2 Disaster Recovery
```python
# recovery_script.py
def restore_from_backup(backup_date):
    # Restore vector database
    restore_vector_store(f"backup/vector_store_{backup_date}")
    
    # Restore conversation history
    restore_conversations(f"backup/conversations_{backup_date}")
    
    # Redeploy application
    subprocess.run(["modal", "deploy", "modal_deployment.py"])
```

## Tahap 8: CI/CD Pipeline

### 8.1 GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy SIPD Chatbot

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install modal
        pip install -r requirements.txt
    
    - name: Deploy to Modal
      env:
        MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
        MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
      run: |
        modal token set --token-id $MODAL_TOKEN_ID --token-secret $MODAL_TOKEN_SECRET
        modal deploy modal_deployment.py
```

### 8.2 Testing Pipeline
```python
# tests/test_chatbot.py
import pytest
import requests

def test_health_endpoint():
    response = requests.get("https://your-app--health.modal.run/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_chat_endpoint():
    payload = {
        "message": "Saya tidak bisa login ke SIPD",
        "session_id": "test_session"
    }
    response = requests.post("https://your-app--chat.modal.run/", json=payload)
    assert response.status_code == 200
    assert "response" in response.json()
```

## Troubleshooting

### Common Issues

1. **Deployment Failed**
   ```bash
   # Check Modal logs
   modal logs your-app-name
   
   # Verify secrets
   modal secret list
   
   # Check resource limits
   modal app describe your-app-name
   ```

2. **High Latency**
   ```python
   # Add performance monitoring
   import time
   
   @app.middleware("http")
   async def add_process_time_header(request, call_next):
       start_time = time.time()
       response = await call_next(request)
       process_time = time.time() - start_time
       response.headers["X-Process-Time"] = str(process_time)
       return response
   ```

3. **Memory Issues**
   ```python
   # Memory optimization
   import gc
   
   def cleanup_memory():
       gc.collect()
       # Clear caches
       cached_response_generation.cache_clear()
   ```

4. **API Rate Limits**
   ```python
   # Implement exponential backoff
   import asyncio
   from tenacity import retry, stop_after_attempt, wait_exponential
   
   @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
   async def call_nebius_api(payload):
       # API call logic
       pass
   ```

## Maintenance Schedule

### Daily
- Monitor health endpoints
- Check error logs
- Verify response times

### Weekly
- Review usage statistics
- Update training data if needed
- Check resource utilization

### Monthly
- Backup vector database
- Review and update model
- Performance optimization
- Security audit

### Quarterly
- Model retraining with new data
- Infrastructure review
- Cost optimization
- Feature updates

## Support & Documentation

### Resources
- [Modal.com Documentation](https://modal.com/docs)
- [Nebius AI Studio Guide](https://studio.nebius.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

### Contact
- Technical Support: [support@yourcompany.com]
- Emergency Hotline: [+62-xxx-xxx-xxxx]
- Documentation: [https://docs.yourcompany.com/sipd-chatbot]

---

**Note**: Panduan ini adalah template umum. Sesuaikan dengan kebutuhan spesifik organisasi Anda.