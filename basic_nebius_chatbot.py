#!/usr/bin/env python3
"""
Basic SIPD Nebius Chatbot
Chatbot paling sederhana yang terhubung dengan Nebius AI
Menggunakan minimal dependencies yang sudah tersedia
"""

import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not available, try to load .env manually
    try:
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except FileNotFoundError:
        pass

# Configuration
class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    sentiment: str
    confidence: float
    suggestions: List[str]
    should_escalate: bool
    metadata: Dict[str, Any]

@dataclass
class BasicConfig:
    """Konfigurasi dasar untuk chatbot"""
    nebius_api_key: str = os.getenv("NEBIUS_API_KEY", "")
    nebius_base_url: str = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1")
    nebius_model_id: str = os.getenv("NEBIUS_MODEL_ID", "meta-llama/Meta-Llama-3.1-70B-Instruct")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "300"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))

class BasicNebiusClient:
    """Client dasar untuk Nebius AI menggunakan requests"""
    
    def __init__(self, config: BasicConfig):
        self.config = config
        
    def generate_response(self, message: str, conversation_history: List[Dict] = None) -> str:
        """Generate response menggunakan Nebius AI"""
        if not self.config.nebius_api_key:
            return "⚠️ Konfigurasi API key Nebius belum diset. Silakan set environment variable NEBIUS_API_KEY."
            
        try:
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": self.get_system_prompt()
                }
            ]
            
            # Add conversation history (last 5 messages only)
            if conversation_history:
                for msg in conversation_history[-5:]:
                    messages.append(msg)
                    
            # Add current message
            messages.append({
                "role": "user",
                "content": message
            })
            
            # API request
            headers = {
                "Authorization": f"Bearer {self.config.nebius_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.config.nebius_model_id,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": False
            }
            
            response = requests.post(
                f"{self.config.nebius_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.request_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                print(f"Nebius API Error {response.status_code}: {response.text}")
                return f"❌ Maaf, terjadi kesalahan saat menghubungi AI (Status: {response.status_code}). Silakan coba lagi atau hubungi administrator."
                
        except requests.exceptions.Timeout:
            return "⏱️ Maaf, response AI timeout. Silakan coba lagi dengan pesan yang lebih singkat."
        except requests.exceptions.ConnectionError:
            return "🌐 Maaf, tidak dapat terhubung ke server AI. Periksa koneksi internet Anda."
        except Exception as e:
            print(f"Error in generate_response: {e}")
            return f"❌ Maaf, terjadi kesalahan: {str(e)}. Silakan hubungi administrator."
            
    def get_system_prompt(self) -> str:
        """Get system prompt untuk SIPD"""
        return """
Anda adalah SIPD Assistant, asisten AI untuk Sistem Informasi Pemerintah Daerah (SIPD).

Tugas Anda:
1. Membantu pengguna dengan masalah SIPD (login, DPA, laporan, teknis)
2. Memberikan solusi yang akurat dan praktis
3. Menunjukkan empati dan profesionalisme
4. Mengarahkan ke sumber daya yang tepat

Guidelines:
- Gunakan bahasa Indonesia yang jelas dan profesional
- Berikan langkah-langkah yang spesifik dan mudah diikuti
- Jika tidak yakin, arahkan ke admin atau dokumentasi resmi
- Selalu konfirmasi pemahaman user sebelum memberikan solusi kompleks
- Tunjukkan empati terhadap frustrasi user
- Jawab dengan singkat tapi informatif (maksimal 3-4 kalimat)

Kontak Support SIPD:
- Email: support@sipd.go.id
- Phone: +62-21-1234567
- Website: https://sipd.kemendagri.go.id

Jawab dengan ramah, helpful, dan profesional. Fokus pada solusi praktis.
"""

class BasicChatEngine:
    """Chat engine dasar"""
    
    def __init__(self, config: BasicConfig):
        self.config = config
        self.nebius_client = BasicNebiusClient(config)
        self.conversations: Dict[str, List[Dict]] = {}
        self.session_stats: Dict[str, Dict] = {}
        
    def classify_intent(self, message: str) -> str:
        """Klasifikasi intent berdasarkan keywords"""
        message_lower = message.lower()
        
        # Login issues
        if any(word in message_lower for word in ['login', 'masuk', 'akses', 'password', 'username', 'sign in']):
            return 'login_issue'
        # DPA issues
        elif any(word in message_lower for word in ['dpa', 'anggaran', 'upload', 'dokumen', 'pelaksanaan']):
            return 'dpa_issue'
        # Report issues
        elif any(word in message_lower for word in ['laporan', 'report', 'export', 'cetak', 'download']):
            return 'laporan_issue'
        # Technical issues
        elif any(word in message_lower for word in ['error', 'gagal', 'tidak bisa', 'bermasalah', 'rusak', 'lambat', 'hang']):
            return 'technical_issue'
        # Greetings
        elif any(word in message_lower for word in ['halo', 'hai', 'selamat', 'terima kasih', 'hello', 'hi']):
            return 'greeting'
        # Complaints
        elif any(word in message_lower for word in ['marah', 'kesal', 'frustasi', 'buruk', 'jelek', 'komplain']):
            return 'complaint'
        else:
            return 'general_inquiry'
            
    def analyze_sentiment(self, message: str) -> str:
        """Analisis sentiment sederhana"""
        message_lower = message.lower()
        
        positive_words = ['bagus', 'baik', 'senang', 'terima kasih', 'mantap', 'hebat', 'puas', 'suka']
        negative_words = ['buruk', 'jelek', 'marah', 'kesal', 'frustasi', 'lambat', 'error', 'gagal', 'benci', 'tidak suka']
        
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
            
    def get_suggestions(self, intent: str) -> List[str]:
        """Get suggestions berdasarkan intent"""
        suggestions_map = {
            'login_issue': [
                'Reset password via "Lupa Password"',
                'Hapus cache browser',
                'Gunakan Chrome/Firefox terbaru',
                'Hubungi admin SIPD'
            ],
            'dpa_issue': [
                'Download template DPA terbaru',
                'Cek format file (.xlsx)',
                'Pastikan kolom wajib terisi',
                'Periksa ukuran file (<10MB)'
            ],
            'laporan_issue': [
                'Refresh halaman laporan',
                'Cek filter tanggal',
                'Coba format export lain',
                'Tunggu jika server sibuk'
            ],
            'technical_issue': [
                'Restart browser',
                'Clear cache & cookies',
                'Coba komputer lain',
                'Screenshot error ke IT'
            ],
            'greeting': [
                'Tanya masalah spesifik',
                'Lihat panduan SIPD',
                'Hubungi support langsung'
            ],
            'complaint': [
                'Sampaikan detail masalah',
                'Hubungi supervisor',
                'Isi form feedback'
            ]
        }
        
        return suggestions_map.get(intent, [
            'Jelaskan masalah lebih detail',
            'Hubungi support untuk bantuan'
        ])
        
    def should_escalate(self, intent: str, sentiment: str, session_id: str) -> bool:
        """Tentukan apakah perlu escalation"""
        # Escalate jika sentiment negatif dan complaint
        if sentiment == 'negative' and intent == 'complaint':
            return True
            
        # Escalate jika user sudah bertanya masalah yang sama berkali-kali
        if session_id in self.session_stats:
            stats = self.session_stats[session_id]
            if stats.get('message_count', 0) >= 5:
                return True
                
        return False
        
    def process_message(self, request: ChatRequest) -> ChatResponse:
        """Process chat message"""
        start_time = time.time()
        
        try:
            # Initialize session if new
            if request.session_id not in self.conversations:
                self.conversations[request.session_id] = []
                self.session_stats[request.session_id] = {
                    'start_time': datetime.now().isoformat(),
                    'message_count': 0
                }
                
            # Update stats
            self.session_stats[request.session_id]['message_count'] += 1
            
            # Classify intent and sentiment
            intent = self.classify_intent(request.message)
            sentiment = self.analyze_sentiment(request.message)
            
            # Get conversation history
            conversation_history = self.conversations[request.session_id]
            
            # Generate response using Nebius
            ai_response = self.nebius_client.generate_response(
                request.message,
                conversation_history
            )
            
            # Get suggestions
            suggestions = self.get_suggestions(intent)
            
            # Check if should escalate
            should_escalate = self.should_escalate(intent, sentiment, request.session_id)
            
            # Update conversation history
            self.conversations[request.session_id].extend([
                {"role": "user", "content": request.message},
                {"role": "assistant", "content": ai_response}
            ])
            
            # Keep only recent messages (last 20)
            if len(self.conversations[request.session_id]) > 20:
                self.conversations[request.session_id] = self.conversations[request.session_id][-20:]
                
            processing_time = time.time() - start_time
                
            return ChatResponse(
                response=ai_response,
                session_id=request.session_id,
                intent=intent,
                sentiment=sentiment,
                confidence=0.8,
                suggestions=suggestions,
                should_escalate=should_escalate,
                metadata={
                    "processing_time": round(processing_time, 2),
                    "model_used": "nebius-ai",
                    "timestamp": datetime.now().isoformat(),
                    "message_count": self.session_stats[request.session_id]['message_count']
                }
            )
            
        except Exception as e:
            print(f"Error processing message: {e}")
            return ChatResponse(
                response=f"❌ Maaf, terjadi kesalahan saat memproses pesan Anda. Silakan coba lagi atau hubungi support.",
                session_id=request.session_id,
                intent="error",
                sentiment="neutral",
                confidence=0.0,
                suggestions=["Coba lagi", "Hubungi support", "Restart browser"],
                should_escalate=True,
                metadata={
                    "processing_time": time.time() - start_time,
                    "model_used": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            )

# FastAPI App
app = FastAPI(
    title="SIPD Basic Nebius Chatbot",
    description="Chatbot AI sederhana untuk SIPD dengan Nebius AI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config = BasicConfig()
chat_engine = BasicChatEngine(config)

@app.get("/", response_class=HTMLResponse)
def get_chat_interface():
    """Serve chat interface"""
    html_content = f"""
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Helpdesk SIPD - AI Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {{
            --primary-gradient: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            --accent-gradient: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            --secondary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --glass-bg: rgba(255, 255, 255, 0.1);
            --glass-border: rgba(255, 255, 255, 0.2);
            --text-primary: #ffffff;
            --text-secondary: rgba(255, 255, 255, 0.8);
            --text-muted: rgba(255, 255, 255, 0.6);
            --success-color: #00ff88;
            --warning-color: #ffaa00;
            --error-color: #ff4757;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--primary-gradient);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            overflow: hidden;
        }}
        
        body::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 80%, rgba(0, 212, 255, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(102, 126, 234, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(118, 75, 162, 0.1) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }}
        
        .chat-container {{
            width: 100%;
            max-width: 900px;
            height: 95vh;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            box-shadow: 
                0 32px 64px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
        }}
        
        .chat-container::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        }}
        
        .chat-header {{
            background: var(--accent-gradient);
            color: var(--text-primary);
            padding: 24px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        
        .chat-header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            animation: shimmer 3s infinite;
        }}
        
        @keyframes shimmer {{
            0% {{ left: -100%; }}
            100% {{ left: 100%; }}
        }}
        
        .header-content {{
            position: relative;
            z-index: 1;
        }}
        
        .chat-header h1 {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }}
        
        .chat-header .logo {{
            font-size: 32px;
            background: linear-gradient(45deg, #ffffff, #e0e0e0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .chat-header p {{
            opacity: 0.9;
            font-size: 16px;
            font-weight: 400;
        }}
        
        .chat-messages {{
            flex: 1;
            padding: 24px;
            overflow-y: auto;
            background: transparent;
            position: relative;
        }}
        
        .chat-messages::-webkit-scrollbar {{
            width: 6px;
        }}
        
        .chat-messages::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }}
        
        .chat-messages::-webkit-scrollbar-thumb {{
            background: var(--accent-gradient);
            border-radius: 3px;
        }}
        
        .message {{
            margin-bottom: 20px;
            display: flex;
            align-items: flex-start;
            gap: 12px;
            animation: messageSlide 0.3s ease-out;
        }}
        
        @keyframes messageSlide {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .message.user {{
            flex-direction: row-reverse;
        }}
        
        .message-content {{
            max-width: 75%;
            padding: 16px 20px;
            border-radius: 20px;
            word-wrap: break-word;
            line-height: 1.5;
            position: relative;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .message.user .message-content {{
            background: var(--accent-gradient);
            color: var(--text-primary);
            border: 1px solid rgba(0, 212, 255, 0.3);
        }}
        
        .message.bot .message-content {{
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
            border: 1px solid var(--glass-border);
        }}
        
        .message-avatar {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            position: relative;
            overflow: hidden;
            flex-shrink: 0;
        }}
        
        .message.user .message-avatar {{
            background: var(--accent-gradient);
            box-shadow: 0 4px 16px rgba(0, 212, 255, 0.3);
        }}
        
        .message.bot .message-avatar {{
            background: var(--secondary-gradient);
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
        }}
        
        .message-avatar::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            animation: avatarShine 2s infinite;
        }}
        
        @keyframes avatarShine {{
            0%, 100% {{ transform: translateX(-100%); }}
            50% {{ transform: translateX(100%); }}
        }}
        
        .suggestions {{
            margin-top: 16px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        
        .suggestion-chip {{
            display: inline-flex;
            align-items: center;
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-secondary);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .suggestion-chip::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: left 0.5s;
        }}
        
        .suggestion-chip:hover {{
            background: var(--accent-gradient);
            color: var(--text-primary);
            border-color: rgba(0, 212, 255, 0.5);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.2);
        }}
        
        .suggestion-chip:hover::before {{
            left: 100%;
        }}
        
        .chat-input {{
            padding: 24px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border-top: 1px solid var(--glass-border);
            position: relative;
        }}
        
        .chat-input::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        }}
        
        .input-container {{
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }}
        
        .message-input {{
            flex: 1;
            padding: 16px 20px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid var(--glass-border);
            border-radius: 25px;
            font-size: 15px;
            color: var(--text-primary);
            outline: none;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
            font-family: inherit;
        }}
        
        .message-input::placeholder {{
            color: var(--text-muted);
        }}
        
        .message-input:focus {{
            border-color: rgba(0, 212, 255, 0.5);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
            background: rgba(255, 255, 255, 0.15);
        }}
        
        .send-button {{
            padding: 16px 24px;
            background: var(--accent-gradient);
            color: var(--text-primary);
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 600;
            font-size: 15px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            min-width: 120px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }}
        
        .send-button::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }}
        
        .send-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
        }}
        
        .send-button:hover::before {{
            left: 100%;
        }}
        
        .send-button:disabled {{
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-muted);
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }}
        
        .typing-indicator {{
            display: none;
            padding: 16px 24px;
            font-style: italic;
            color: var(--text-muted);
            font-size: 14px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            margin: 8px 0;
            backdrop-filter: blur(10px);
        }}
        
        .status-bar {{
            padding: 12px 24px;
            background: rgba(255, 255, 255, 0.05);
            font-size: 12px;
            color: var(--text-muted);
            border-top: 1px solid var(--glass-border);
            backdrop-filter: blur(20px);
        }}
        
        .config-info {{
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 12px;
            padding: 16px;
            margin: 16px 0;
            font-size: 13px;
            color: var(--text-secondary);
            backdrop-filter: blur(10px);
        }}
        
        .error-message {{
            background: rgba(255, 71, 87, 0.1);
            border: 1px solid rgba(255, 71, 87, 0.3);
            border-radius: 12px;
            padding: 16px;
            margin: 16px 0;
            color: var(--error-color);
            backdrop-filter: blur(10px);
        }}
        
        .pulse {{
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
            }}
            50% {{
                opacity: 0.7;
            }}
        }}
        
        .glow {{
            animation: glow 2s ease-in-out infinite alternate;
        }}
        
        @keyframes glow {{
            from {{
                box-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
            }}
            to {{
                box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
            }}
        }}
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div class="header-content">
                <h1>
                    <i class="fas fa-robot logo"></i>
                    Helpdesk SIPD
                </h1>
                <p>AI Assistant untuk Sistem Informasi Pemerintah Daerah</p>
            </div>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message bot">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    Selamat datang di Helpdesk SIPD! 👋<br><br>
                    Saya adalah AI Assistant yang siap membantu Anda dengan berbagai masalah SIPD seperti:
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>🔐 Login dan akses sistem</li>
                        <li>📄 DPA (Dokumen Pelaksanaan Anggaran)</li>
                        <li>📊 Laporan dan pelaporan</li>
                        <li>🔧 Masalah teknis sistem</li>
                        <li>💡 Panduan dan tutorial</li>
                    </ul>
                    
                    {'<div class="error-message">⚠️ <strong>API Key belum diset!</strong><br>Set environment variable NEBIUS_API_KEY untuk menggunakan AI.</div>' if not config.nebius_api_key else ''}
                    
                    <div class="suggestions">
                        <span class="suggestion-chip" onclick="sendMessage('Saya tidak bisa login ke SIPD')">🔐 Login Issue</span>
                        <span class="suggestion-chip" onclick="sendMessage('Bagaimana cara upload DPA?')">📄 Upload DPA</span>
                        <span class="suggestion-chip" onclick="sendMessage('Laporan tidak muncul')">📊 Masalah Laporan</span>
                        <span class="suggestion-chip" onclick="sendMessage('Sistem error terus')">🔧 Error Teknis</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            🤖 Sedang mengetik...
        </div>
        
        <div class="chat-input">
            <div class="input-container">
                <input type="text" id="messageInput" class="message-input" 
                       placeholder="💬 Tanyakan sesuatu tentang SIPD..." 
                       onkeypress="handleKeyPress(event)">
                <button id="sendButton" class="send-button" onclick="sendMessage()">
                    <i class="fas fa-paper-plane"></i>
                    Kirim
                </button>
            </div>
        </div>
        
        <div class="status-bar" id="statusBar" style="display: none;">
            Status: Siap
        </div>
    </div>

    <script>
        const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        let messageCount = 0;
        
        function addMessage(content, isUser = false, suggestions = []) {{
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${{isUser ? 'user' : 'bot'}}`;
            
            let suggestionsHtml = '';
            if (suggestions.length > 0) {{
                suggestionsHtml = '<div class="suggestions">';
                suggestions.forEach(suggestion => {{
                    const escapedSuggestion = suggestion.replace(/'/g, "\\'").replace(/"/g, '&quot;');
                    suggestionsHtml += `<span class="suggestion-chip" onclick="sendMessage('${{escapedSuggestion}}')">💡 ${{suggestion}}</span>`;
                }});
                suggestionsHtml += '</div>';
            }}
            
            const avatarIcon = isUser ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
            
            messageDiv.innerHTML = `
                <div class="message-avatar">
                    ${{avatarIcon}}
                </div>
                <div class="message-content">
                    ${{content}}
                    ${{suggestionsHtml}}
                </div>
            `;
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }}
        
        function showTyping() {{
            document.getElementById('typingIndicator').style.display = 'block';
        }}
        
        function hideTyping() {{
            document.getElementById('typingIndicator').style.display = 'none';
        }}
        
        function updateStatus(message) {{
            document.getElementById('statusBar').textContent = message;
        }}
        
        function sendMessage(message = null) {{
            const input = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            
            const messageText = message || input.value.trim();
            if (!messageText) return;
            
            // Add user message
            addMessage(messageText, true);
            
            // Clear input and disable button
            input.value = '';
            sendButton.disabled = true;
            showTyping();
            updateStatus('Mengirim pesan ke Nebius AI...');
            
            // Send to API
            fetch('/chat', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify({{
                    message: messageText,
                    session_id: sessionId,
                    context: {{
                        browser: navigator.userAgent,
                        timestamp: new Date().toISOString(),
                        screen_resolution: screen.width + 'x' + screen.height
                    }}
                }})
            }})
            .then(response => {{
                if (!response.ok) {{
                    throw new Error(`HTTP ${{response.status}}`);
                }}
                return response.json();
            }})
            .then(data => {{
                // Add bot response
                addMessage(data.response, false, data.suggestions);
                
                // Update status
                messageCount++;
                const processingTime = data.metadata.processing_time || 0;
                updateStatus(`Pesan #${{messageCount}} | Intent: ${{data.intent}} | Sentiment: ${{data.sentiment}} | Time: ${{processingTime}}s | Nebius AI ✅`);
                
                // Show escalation warning if needed
                if (data.should_escalate) {{
                    setTimeout(() => {{
                        addMessage('⚠️ <strong>Escalation Alert:</strong> Sepertinya Anda membutuhkan bantuan lebih lanjut. Tim support akan segera menghubungi Anda atau silakan hubungi langsung ke support@sipd.go.id', false);
                    }}, 1000);
                }}
            }})
            .catch(error => {{
                console.error('Error:', error);
                addMessage(`❌ <strong>Error:</strong> ${{error.message}}<br><br>Kemungkinan penyebab:<br>• Koneksi internet bermasalah<br>• Server sedang sibuk<br>• API key Nebius belum diset<br><br>Silakan coba lagi atau hubungi administrator.`, false, ['Coba lagi', 'Hubungi support']);
                updateStatus('Error dalam mengirim pesan');
            }})
            .finally(() => {{
                hideTyping();
                sendButton.disabled = false;
                input.focus();
            }});
        }}
        
        function handleKeyPress(event) {{
            if (event.key === 'Enter') {{
                sendMessage();
            }}
        }}
        
        // Focus input on load
        document.addEventListener('DOMContentLoaded', function() {{
            document.getElementById('messageInput').focus();
        }});
        
        // Auto-scroll to bottom
        function scrollToBottom() {{
            const messagesContainer = document.getElementById('chatMessages');
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }}
        
        // Call scroll on load
        window.addEventListener('load', scrollToBottom);
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        response = chat_engine.process_message(request)
        return response
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Test Nebius connection with a simple request
        if config.nebius_api_key:
            test_response = chat_engine.nebius_client.generate_response("test")
            nebius_ok = "error" not in test_response.lower() and len(test_response) > 0
        else:
            nebius_ok = False
    except:
        nebius_ok = False
        
    return {
        "status": "healthy" if nebius_ok else "degraded",
        "nebius_connection": nebius_ok,
        "api_key_configured": bool(config.nebius_api_key),
        "active_sessions": len(chat_engine.conversations),
        "total_conversations": sum(len(conv) for conv in chat_engine.conversations.values()),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.nebius_model_id,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "timeout": config.request_timeout
        }
    }

@app.get("/stats")
def get_stats():
    """Get chatbot statistics"""
    total_sessions = len(chat_engine.conversations)
    total_messages = sum(len(conv) for conv in chat_engine.conversations.values())
    
    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "avg_messages_per_session": round(total_messages / total_sessions, 2) if total_sessions > 0 else 0,
        "active_sessions": total_sessions,
        "api_key_configured": bool(config.nebius_api_key),
        "model_info": {
            "model_id": config.nebius_model_id,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/chat/history/{session_id}")
def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if session_id in chat_engine.conversations:
        return {
            "session_id": session_id,
            "messages": chat_engine.conversations[session_id],
            "stats": chat_engine.session_stats.get(session_id, {}),
            "message_count": len(chat_engine.conversations[session_id])
        }
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.delete("/chat/history/{session_id}")
def clear_conversation_history(session_id: str):
    """Clear conversation history for a session"""
    if session_id in chat_engine.conversations:
        del chat_engine.conversations[session_id]
        if session_id in chat_engine.session_stats:
            del chat_engine.session_stats[session_id]
        return {"message": f"Conversation history for session {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    print("🚀 Starting SIPD Basic Nebius Chatbot...")
    print("=" * 50)
    print(f"📡 Nebius API: {config.nebius_base_url}")
    print(f"🤖 Model: {config.nebius_model_id}")
    print(f"🔑 API Key: {'✅ Configured' if config.nebius_api_key else '❌ Not set (set NEBIUS_API_KEY env var)'}")
    print(f"⚙️ Max Tokens: {config.max_tokens} | Temperature: {config.temperature}")
    print(f"⏱️ Timeout: {config.request_timeout}s")
    print()
    print("🌐 Access the chatbot at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    print("❤️ Health check at: http://localhost:8000/health")
    print("📊 Statistics at: http://localhost:8000/stats")
    print()
    if not config.nebius_api_key:
        print("⚠️ WARNING: NEBIUS_API_KEY not set!")
        print("   Set it with: set NEBIUS_API_KEY=your_api_key_here")
        print("   Or create .env file with: NEBIUS_API_KEY=your_api_key_here")
        print()
    print("=" * 50)
    
    uvicorn.run(
        "basic_nebius_chatbot:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )