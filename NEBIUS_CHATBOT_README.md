# 🤖 SIPD Nebius Chatbot

Chatbot AI canggih untuk Sistem Informasi Pemerintah Daerah (SIPD) yang terhubung langsung dengan **Nebius AI Studio** untuk response generation yang intelligent dan natural.

## ✨ Fitur Utama

### 🧠 **AI-Powered Intelligence**
- **Nebius AI Integration**: Terhubung langsung dengan model AI terbaru di Nebius AI Studio
- **Advanced NLP**: Pemahaman konteks dan intent yang akurat
- **Sentiment Analysis**: Deteksi emosi user untuk response yang empati
- **Context Awareness**: Mempertahankan konteks percakapan untuk dialog yang natural

### 🎯 **SIPD-Specific Features**
- **Domain Expertise**: Khusus untuk masalah SIPD (login, DPA, laporan, teknis)
- **Intelligent Escalation**: Otomatis mengarahkan ke human agent jika diperlukan
- **Solution Database**: Database solusi terintegrasi untuk masalah umum
- **Multi-Intent Support**: Menangani berbagai jenis pertanyaan dalam satu percakapan

### 🚀 **Advanced Capabilities**
- **RAG System**: Retrieval-Augmented Generation dengan ChromaDB
- **Embedding Search**: Pencarian semantik untuk konteks yang relevan
- **Conversation Memory**: Menyimpan dan menganalisis riwayat percakapan
- **Performance Monitoring**: Metrics dan analytics real-time

### 🛡️ **Enterprise-Ready**
- **Security**: Input validation, rate limiting, dan sanitization
- **Scalability**: Async architecture untuk high concurrency
- **Monitoring**: Health checks, metrics, dan alerting
- **Configuration**: Flexible configuration untuk berbagai environment

## 🏗️ Arsitektur

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │────│   FastAPI App   │────│  Nebius Client  │
│   (Browser)     │    │   (nebius_      │    │   (AI Model)    │
│                 │    │    chatbot.py)  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   RAG System    │
                       │   (ChromaDB +   │
                       │   Embeddings)   │
                       └─────────────────┘
```

## 🚀 Quick Start

### 1. **Environment Setup**

```bash
# Clone atau pastikan Anda di direktori ChatbotAI
cd ChatbotAI

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 2. **Konfigurasi Nebius AI**

Edit file `.env` dan tambahkan konfigurasi Nebius:

```env
# Nebius AI Configuration
NEBIUS_API_KEY=your_nebius_api_key_here
NEBIUS_BASE_URL=https://api.studio.nebius.ai/v1
NEBIUS_MODEL_ID=meta-llama/Meta-Llama-3.1-70B-Instruct
NEBIUS_EMBEDDING_MODEL=BAAI/bge-m3

# Chatbot Configuration
MAX_TOKENS=500
TEMPERATURE=0.7
RAG_ENABLED=true
ENABLE_CACHING=true
ENABLE_RATE_LIMITING=true

# Security
RATE_LIMIT_PER_MINUTE=60
MAX_MESSAGE_LENGTH=2000
```

### 3. **Jalankan Chatbot**

```bash
# Jalankan dengan script launcher
python run_nebius_chatbot.py

# Atau dengan opsi khusus
python run_nebius_chatbot.py --port 8080 --reload

# Test connection saja
python run_nebius_chatbot.py --test
```

### 4. **Akses Chatbot**

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Statistics**: http://localhost:8000/stats

## 📋 API Endpoints

### **Chat Endpoints**

```http
POST /chat
Content-Type: application/json

{
  "message": "Saya tidak bisa login ke SIPD",
  "session_id": "user_session_123",
  "user_id": "optional_user_id",
  "context": {
    "browser": "Chrome",
    "os": "Windows 10"
  }
}
```

**Response:**
```json
{
  "response": "Saya akan membantu Anda dengan masalah login...",
  "session_id": "user_session_123",
  "intent": "login_issue",
  "sentiment": "neutral",
  "confidence": 0.85,
  "suggestions": [
    "Coba reset password",
    "Hapus cache browser",
    "Hubungi admin SIPD"
  ],
  "should_escalate": false,
  "metadata": {
    "processing_time": 1.23,
    "model_used": "nebius-ai",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### **Management Endpoints**

```http
# Get conversation history
GET /chat/history/{session_id}

# Clear conversation history
DELETE /chat/history/{session_id}

# Get chatbot statistics
GET /stats

# Health check
GET /health
```

## 🎛️ Konfigurasi

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `NEBIUS_API_KEY` | - | **Required**: API key untuk Nebius AI |
| `NEBIUS_MODEL_ID` | `meta-llama/Meta-Llama-3.1-70B-Instruct` | Model ID di Nebius |
| `MAX_TOKENS` | `500` | Maksimal token untuk response |
| `TEMPERATURE` | `0.7` | Kreativitas response (0.0-2.0) |
| `RAG_ENABLED` | `true` | Enable RAG system |
| `ENABLE_CACHING` | `true` | Enable response caching |
| `RATE_LIMIT_PER_MINUTE` | `60` | Rate limit per user |

### **Advanced Configuration**

Edit `nebius_chatbot_config.py` untuk konfigurasi advanced:

```python
# Performance tuning
max_concurrent_requests: int = 10
request_timeout: int = 30
cache_ttl: int = 3600

# RAG configuration
rag_similarity_threshold: float = 0.7
rag_max_context_length: int = 1000
rag_top_k_results: int = 5

# Escalation rules
escalation_sentiment_threshold: float = -0.5
escalation_repeated_issues: int = 3
```

## 🧪 Testing

### **Connection Test**
```bash
# Test Nebius connection
python run_nebius_chatbot.py --test
```

### **Manual Testing**
```bash
# Test dengan curl
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Halo, saya butuh bantuan dengan SIPD",
       "session_id": "test_session"
     }'
```

### **Load Testing**
```bash
# Install artillery untuk load testing
npm install -g artillery

# Run load test
artillery quick --count 10 --num 5 http://localhost:8000/health
```

## 📊 Monitoring

### **Health Monitoring**
```bash
# Check health status
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "nebius_connection": true,
  "rag_system": true,
  "active_sessions": 5,
  "total_conversations": 127,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### **Usage Statistics**
```bash
# Get usage stats
curl http://localhost:8000/stats
```

**Response:**
```json
{
  "total_sessions": 50,
  "total_messages": 250,
  "avg_messages_per_session": 5.0,
  "intent_distribution": {
    "login_issue": 45,
    "dpa_issue": 30,
    "technical_issue": 25
  },
  "sentiment_distribution": {
    "positive": 60,
    "neutral": 80,
    "negative": 10
  }
}
```

## 🔧 Development

### **Development Mode**
```bash
# Run dengan auto-reload
python run_nebius_chatbot.py --reload --verbose
```

### **Custom Prompts**
Edit prompts di `nebius_chatbot_config.py`:

```python
class PromptTemplates:
    SYSTEM_PROMPT_BASE = """
    Customize your system prompt here...
    """
    
    RESPONSE_GENERATION_PROMPT = """
    Customize response generation...
    """
```

### **Adding New Intents**
```python
# Di nebius_chatbot_config.py
common_solutions = {
    "new_intent": [
        "Solution 1",
        "Solution 2",
        "Solution 3"
    ]
}
```

## 🚀 Deployment

### **Production Deployment**

1. **Environment Setup**
```bash
# Production environment
export NEBIUS_API_KEY="your_production_key"
export LOG_LEVEL="INFO"
export ENABLE_METRICS="true"
```

2. **Run with Gunicorn**
```bash
# Install gunicorn
pip install gunicorn

# Run production server
gunicorn nebius_chatbot:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

3. **Docker Deployment**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "run_nebius_chatbot.py", "--host", "0.0.0.0"]
```

### **Modal.com Deployment**
Gunakan `modal_deployment.py` yang sudah ada untuk deploy ke Modal.com.

## 🔍 Troubleshooting

### **Common Issues**

**1. Nebius Connection Failed**
```bash
# Check API key
echo $NEBIUS_API_KEY

# Test connection
python run_nebius_chatbot.py --test
```

**2. Slow Response Times**
```python
# Reduce max_tokens
MAX_TOKENS=200

# Enable caching
ENABLE_CACHING=true

# Increase timeout
REQUEST_TIMEOUT=60
```

**3. Memory Issues**
```python
# Reduce conversation history
MAX_CONVERSATION_HISTORY=20

# Reduce context window
CONTEXT_WINDOW_SIZE=3
```

**4. Rate Limiting**
```python
# Adjust rate limits
RATE_LIMIT_PER_MINUTE=30
MAX_CONCURRENT_REQUESTS=5
```

### **Debug Mode**
```bash
# Run dengan verbose logging
python run_nebius_chatbot.py --verbose

# Check logs
tail -f logs/chatbot.log
```

## 📈 Performance Optimization

### **Caching Strategy**
- **Response Caching**: Cache responses untuk pertanyaan serupa
- **Embedding Caching**: Cache embeddings untuk teks yang sama
- **RAG Caching**: Cache hasil pencarian RAG

### **Connection Pooling**
- HTTP connection pooling untuk Nebius API
- Database connection pooling untuk ChromaDB
- Async request handling

### **Resource Management**
- Memory-efficient conversation storage
- Automatic cleanup untuk old sessions
- Configurable resource limits

## 🤝 Contributing

1. **Fork** repository ini
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** Pull Request

## 📄 License

Project ini menggunakan MIT License. Lihat file `LICENSE` untuk detail.

## 🆘 Support

- **Documentation**: [Nebius AI Studio Docs](https://docs.nebius.ai/)
- **Issues**: Buat issue di repository ini
- **Email**: support@sipd.go.id
- **Discord**: [SIPD Community](https://discord.gg/sipd)

---

**Made with ❤️ for SIPD Community**

*Powered by Nebius AI Studio & FastAPI*