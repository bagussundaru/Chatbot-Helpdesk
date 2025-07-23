#!/usr/bin/env python3
"""
Simple SIPD Nebius Chatbot
Chatbot sederhana yang terhubung dengan Nebius AI tanpa dependencies kompleks
"""

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

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
class SimpleConfig:
    """Konfigurasi sederhana untuk chatbot"""
    nebius_api_key: str = os.getenv("NEBIUS_API_KEY", "")
    nebius_base_url: str = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1")
    nebius_model_id: str = os.getenv("NEBIUS_MODEL_ID", "meta-llama/Meta-Llama-3.1-70B-Instruct")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "500"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    max_conversation_history: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))

class SimpleNebiusClient:
    """Client sederhana untuk Nebius AI"""
    
    def __init__(self, config: SimpleConfig):
        self.config = config
        self.session = None
        
    async def get_session(self):
        """Get atau create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
        
    async def generate_response(self, message: str, conversation_history: List[Dict] = None) -> str:
        """Generate response menggunakan Nebius AI"""
        if not self.config.nebius_api_key:
            return "Maaf, konfigurasi API key Nebius belum diset. Silakan hubungi administrator."
            
        try:
            session = await self.get_session()
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": self.get_system_prompt()
                }
            ]
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-10:]:  # Last 10 messages
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
            
            async with session.post(
                f"{self.config.nebius_base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    error_text = await response.text()
                    print(f"Nebius API Error {response.status}: {error_text}")
                    return f"Maaf, terjadi kesalahan saat menghubungi AI. Status: {response.status}"
                    
        except asyncio.TimeoutError:
            return "Maaf, response AI timeout. Silakan coba lagi."
        except Exception as e:
            print(f"Error in generate_response: {e}")
            return f"Maaf, terjadi kesalahan: {str(e)}"
            
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

Kontak Support:
- Email: support@sipd.go.id
- Phone: +62-21-1234567
- Website: https://sipd.kemendagri.go.id

Jawab dengan ramah, helpful, dan profesional.
"""

    async def close(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

class SimpleChatEngine:
    """Chat engine sederhana"""
    
    def __init__(self, config: SimpleConfig):
        self.config = config
        self.nebius_client = SimpleNebiusClient(config)
        self.conversations: Dict[str, List[Dict]] = {}
        self.session_stats: Dict[str, Dict] = {}
        
    def classify_intent(self, message: str) -> str:
        """Klasifikasi intent sederhana berdasarkan keywords"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['login', 'masuk', 'akses', 'password', 'username']):
            return 'login_issue'
        elif any(word in message_lower for word in ['dpa', 'anggaran', 'upload', 'dokumen']):
            return 'dpa_issue'
        elif any(word in message_lower for word in ['laporan', 'report', 'export', 'cetak']):
            return 'laporan_issue'
        elif any(word in message_lower for word in ['error', 'gagal', 'tidak bisa', 'bermasalah', 'rusak']):
            return 'technical_issue'
        elif any(word in message_lower for word in ['halo', 'hai', 'selamat', 'terima kasih']):
            return 'greeting'
        elif any(word in message_lower for word in ['marah', 'kesal', 'frustasi', 'lambat', 'buruk']):
            return 'complaint'
        else:
            return 'general_inquiry'
            
    def analyze_sentiment(self, message: str) -> str:
        """Analisis sentiment sederhana"""
        message_lower = message.lower()
        
        positive_words = ['bagus', 'baik', 'senang', 'terima kasih', 'mantap', 'hebat']
        negative_words = ['buruk', 'jelek', 'marah', 'kesal', 'frustasi', 'lambat', 'error', 'gagal']
        
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
                'Reset password melalui menu "Lupa Password"',
                'Hapus cache dan cookies browser',
                'Coba gunakan browser lain (Chrome/Firefox)',
                'Hubungi admin jika masalah berlanjut'
            ],
            'dpa_issue': [
                'Download template DPA terbaru',
                'Periksa format file (.xlsx atau .xls)',
                'Pastikan semua kolom wajib terisi',
                'Cek ukuran file (maksimal 10MB)'
            ],
            'laporan_issue': [
                'Refresh halaman laporan',
                'Periksa filter tanggal yang dipilih',
                'Coba export dalam format berbeda',
                'Tunggu beberapa saat jika server sibuk'
            ],
            'technical_issue': [
                'Restart browser dan coba lagi',
                'Clear cache dan cookies',
                'Coba dari komputer/jaringan lain',
                'Laporkan ke tim IT dengan screenshot'
            ],
            'greeting': [
                'Tanyakan masalah spesifik yang Anda hadapi',
                'Lihat panduan penggunaan SIPD',
                'Hubungi support jika butuh bantuan langsung'
            ]
        }
        
        return suggestions_map.get(intent, [
            'Jelaskan masalah Anda lebih detail',
            'Hubungi support untuk bantuan lebih lanjut'
        ])
        
    def should_escalate(self, intent: str, sentiment: str, session_id: str) -> bool:
        """Tentukan apakah perlu escalation"""
        # Escalate jika sentiment sangat negatif
        if sentiment == 'negative':
            return True
            
        # Escalate jika user sudah bertanya masalah yang sama berkali-kali
        if session_id in self.session_stats:
            stats = self.session_stats[session_id]
            if stats.get('repeated_issues', 0) >= 3:
                return True
                
        return False
        
    async def process_message(self, request: ChatRequest) -> ChatResponse:
        """Process chat message"""
        try:
            # Initialize session if new
            if request.session_id not in self.conversations:
                self.conversations[request.session_id] = []
                self.session_stats[request.session_id] = {
                    'start_time': datetime.now().isoformat(),
                    'message_count': 0,
                    'repeated_issues': 0
                }
                
            # Update stats
            self.session_stats[request.session_id]['message_count'] += 1
            
            # Classify intent and sentiment
            intent = self.classify_intent(request.message)
            sentiment = self.analyze_sentiment(request.message)
            
            # Get conversation history
            conversation_history = self.conversations[request.session_id]
            
            # Generate response using Nebius
            ai_response = await self.nebius_client.generate_response(
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
            
            # Keep only recent messages
            if len(self.conversations[request.session_id]) > self.config.max_conversation_history:
                self.conversations[request.session_id] = self.conversations[request.session_id][-self.config.max_conversation_history:]
                
            return ChatResponse(
                response=ai_response,
                session_id=request.session_id,
                intent=intent,
                sentiment=sentiment,
                confidence=0.8,  # Static confidence for simplicity
                suggestions=suggestions,
                should_escalate=should_escalate,
                metadata={
                    "processing_time": 1.0,  # Placeholder
                    "model_used": "nebius-ai",
                    "timestamp": datetime.now().isoformat(),
                    "message_count": self.session_stats[request.session_id]['message_count']
                }
            )
            
        except Exception as e:
            print(f"Error processing message: {e}")
            return ChatResponse(
                response=f"Maaf, terjadi kesalahan saat memproses pesan Anda: {str(e)}",
                session_id=request.session_id,
                intent="error",
                sentiment="neutral",
                confidence=0.0,
                suggestions=["Coba lagi dalam beberapa saat", "Hubungi support jika masalah berlanjut"],
                should_escalate=True,
                metadata={
                    "processing_time": 0.0,
                    "model_used": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            )

# FastAPI App
app = FastAPI(
    title="SIPD Nebius Chatbot",
    description="Chatbot AI untuk Sistem Informasi Pemerintah Daerah dengan Nebius AI",
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
config = SimpleConfig()
chat_engine = SimpleChatEngine(config)

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve chat interface"""
    html_content = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIPD Nebius Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .chat-header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .chat-header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
        }
        
        .message.user .message-content {
            background: #667eea;
            color: white;
        }
        
        .message.bot .message-content {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
        }
        
        .suggestions {
            margin-top: 10px;
        }
        
        .suggestion-chip {
            display: inline-block;
            background: #e3f2fd;
            color: #1976d2;
            padding: 6px 12px;
            margin: 4px;
            border-radius: 16px;
            font-size: 12px;
            cursor: pointer;
            border: 1px solid #bbdefb;
            transition: all 0.2s;
        }
        
        .suggestion-chip:hover {
            background: #1976d2;
            color: white;
        }
        
        .chat-input {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
        }
        
        .message-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        
        .message-input:focus {
            border-color: #667eea;
        }
        
        .send-button {
            padding: 12px 24px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.2s;
        }
        
        .send-button:hover {
            background: #5a6fd8;
        }
        
        .send-button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .typing-indicator {
            display: none;
            padding: 10px;
            font-style: italic;
            color: #666;
        }
        
        .status-bar {
            padding: 10px 20px;
            background: #f0f0f0;
            font-size: 12px;
            color: #666;
            border-top: 1px solid #e0e0e0;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🤖 SIPD Nebius Chatbot</h1>
            <p>Asisten AI untuk Sistem Informasi Pemerintah Daerah</p>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message bot">
                <div class="message-content">
                    Selamat datang di SIPD Chatbot! 👋<br>
                    Saya siap membantu Anda dengan masalah SIPD seperti login, DPA, laporan, dan masalah teknis lainnya.
                    <div class="suggestions">
                        <span class="suggestion-chip" onclick="sendMessage('Saya tidak bisa login ke SIPD')">Login Issue</span>
                        <span class="suggestion-chip" onclick="sendMessage('Bagaimana cara upload DPA?')">Upload DPA</span>
                        <span class="suggestion-chip" onclick="sendMessage('Laporan tidak muncul')">Masalah Laporan</span>
                        <span class="suggestion-chip" onclick="sendMessage('Sistem error terus')">Error Teknis</span>
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
                       placeholder="Ketik pesan Anda di sini..." 
                       onkeypress="handleKeyPress(event)">
                <button id="sendButton" class="send-button" onclick="sendMessage()">Kirim</button>
            </div>
        </div>
        
        <div class="status-bar" id="statusBar">
            Status: Terhubung dengan Nebius AI ✅
        </div>
    </div>

    <script>
        const sessionId = 'session_' + Date.now();
        let messageCount = 0;
        
        function addMessage(content, isUser = false, suggestions = []) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            
            let suggestionsHtml = '';
            if (suggestions.length > 0) {
                suggestionsHtml = '<div class="suggestions">';
                suggestions.forEach(suggestion => {
                    suggestionsHtml += `<span class="suggestion-chip" onclick="sendMessage('${suggestion.replace(/'/g, "\\'")}')">💡 ${suggestion}</span>`;
                });
                suggestionsHtml += '</div>';
            }
            
            messageDiv.innerHTML = `
                <div class="message-content">
                    ${content}
                    ${suggestionsHtml}
                </div>
            `;
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function showTyping() {
            document.getElementById('typingIndicator').style.display = 'block';
        }
        
        function hideTyping() {
            document.getElementById('typingIndicator').style.display = 'none';
        }
        
        function updateStatus(message) {
            document.getElementById('statusBar').textContent = message;
        }
        
        async function sendMessage(message = null) {
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
            updateStatus('Mengirim pesan...');
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: messageText,
                        session_id: sessionId,
                        context: {
                            browser: navigator.userAgent,
                            timestamp: new Date().toISOString()
                        }
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    
                    // Add bot response
                    addMessage(data.response, false, data.suggestions);
                    
                    // Update status
                    messageCount++;
                    updateStatus(`Pesan ke-${messageCount} | Intent: ${data.intent} | Sentiment: ${data.sentiment} | Nebius AI ✅`);
                    
                    // Show escalation warning if needed
                    if (data.should_escalate) {
                        setTimeout(() => {
                            addMessage('⚠️ Sepertinya Anda membutuhkan bantuan lebih lanjut. Tim support akan segera menghubungi Anda.', false);
                        }, 1000);
                    }
                } else {
                    const errorData = await response.json();
                    addMessage(`❌ Error: ${errorData.detail || 'Terjadi kesalahan'}`, false);
                    updateStatus('Error dalam mengirim pesan');
                }
            } catch (error) {
                addMessage('❌ Koneksi bermasalah. Silakan coba lagi.', false);
                updateStatus('Koneksi bermasalah');
                console.error('Error:', error);
            } finally {
                hideTyping();
                sendButton.disabled = false;
                input.focus();
            }
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        // Focus input on load
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('messageInput').focus();
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        response = await chat_engine.process_message(request)
        return response
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Nebius connection
        test_response = await chat_engine.nebius_client.generate_response("test", [])
        nebius_ok = len(test_response) > 0
    except:
        nebius_ok = False
        
    return {
        "status": "healthy" if nebius_ok else "degraded",
        "nebius_connection": nebius_ok,
        "active_sessions": len(chat_engine.conversations),
        "total_conversations": sum(len(conv) for conv in chat_engine.conversations.values()),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.nebius_model_id,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature
        }
    }

@app.get("/stats")
async def get_stats():
    """Get chatbot statistics"""
    total_sessions = len(chat_engine.conversations)
    total_messages = sum(len(conv) for conv in chat_engine.conversations.values())
    
    # Calculate intent distribution
    intent_counts = {}
    for session_id, stats in chat_engine.session_stats.items():
        # This is simplified - in real implementation you'd track intents per message
        pass
        
    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "avg_messages_per_session": total_messages / total_sessions if total_sessions > 0 else 0,
        "active_sessions": total_sessions,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/chat/history/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if session_id in chat_engine.conversations:
        return {
            "session_id": session_id,
            "messages": chat_engine.conversations[session_id],
            "stats": chat_engine.session_stats.get(session_id, {})
        }
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.delete("/chat/history/{session_id}")
async def clear_conversation_history(session_id: str):
    """Clear conversation history for a session"""
    if session_id in chat_engine.conversations:
        del chat_engine.conversations[session_id]
        if session_id in chat_engine.session_stats:
            del chat_engine.session_stats[session_id]
        return {"message": "Conversation history cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await chat_engine.nebius_client.close()

if __name__ == "__main__":
    print("🚀 Starting SIPD Nebius Chatbot...")
    print(f"📡 Nebius API: {config.nebius_base_url}")
    print(f"🤖 Model: {config.nebius_model_id}")
    print(f"🔑 API Key: {'✅ Set' if config.nebius_api_key else '❌ Not set'}")
    print("\n🌐 Access the chatbot at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    print("❤️ Health check at: http://localhost:8000/health")
    print("\n" + "="*50)
    
    uvicorn.run(
        "simple_nebius_chatbot:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )