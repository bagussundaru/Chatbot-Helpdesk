# SIPD Chatbot dengan Integrasi Nebius AI
# Chatbot yang terhubung langsung dengan model AI di Nebius untuk response generation

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# Import konfigurasi dan client Nebius
from enhanced_config import EnhancedConfig
from nebius_client import NebiusClient
from nebius_embedding_integration import NebiusEmbeddingClient, EnhancedRAGSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models untuk API
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Pesan dari user")
    session_id: str = Field(..., min_length=1, max_length=100, description="ID sesi chat")
    user_id: Optional[str] = Field(None, description="ID user (opsional)")
    context: Optional[Dict[str, Any]] = Field(None, description="Konteks tambahan")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Response dari chatbot")
    session_id: str = Field(..., description="ID sesi chat")
    intent: Optional[str] = Field(None, description="Intent yang terdeteksi")
    sentiment: Optional[str] = Field(None, description="Sentiment yang terdeteksi")
    confidence: Optional[float] = Field(None, description="Confidence score")
    suggestions: List[str] = Field(default_factory=list, description="Saran tindak lanjut")
    should_escalate: bool = Field(False, description="Apakah perlu eskalasi ke human agent")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata tambahan")

class NebiusChatbot:
    """Chatbot yang terhubung dengan Nebius AI untuk response generation yang intelligent"""
    
    def __init__(self):
        self.config = EnhancedConfig()
        self.nebius_client = None
        self.embedding_client = None
        self.rag_system = None
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.user_profiles: Dict[str, Dict] = {}
        
        # Template prompts untuk berbagai skenario SIPD
        self.system_prompts = {
            "default": """
Anda adalah asisten AI untuk Sistem Informasi Pemerintah Daerah (SIPD). 
Anda membantu pengguna dengan masalah terkait:
- Login dan akses sistem
- Upload dan pengelolaan DPA (Dokumen Pelaksanaan Anggaran)
- Pembuatan dan akses laporan
- Masalah teknis sistem
- Panduan penggunaan fitur SIPD

Berikan jawaban yang:
- Akurat dan berdasarkan pengetahuan SIPD
- Ramah dan profesional
- Langkah-langkah yang jelas jika diperlukan
- Empati terhadap masalah user

Jika tidak yakin dengan jawaban, sarankan untuk menghubungi admin SIPD.
""",
            "login_issue": """
Anda sedang membantu user dengan masalah login SIPD. 
Berikan solusi step-by-step yang jelas dan tanyakan detail spesifik jika diperlukan.
Pastikan untuk menanyakan:
- Browser yang digunakan
- Pesan error yang muncul
- Apakah password sudah dicoba direset
""",
            "dpa_issue": """
Anda membantu dengan masalah DPA (Dokumen Pelaksanaan Anggaran).
Fokus pada:
- Format file yang benar
- Proses upload yang tepat
- Validasi data DPA
- Status persetujuan
""",
            "laporan_issue": """
Anda membantu dengan masalah laporan SIPD.
Bantu user dengan:
- Cara mengakses laporan
- Filter dan parameter laporan
- Export laporan
- Troubleshooting tampilan laporan
"""
        }
    
    async def initialize(self):
        """Inisialisasi semua komponen chatbot"""
        try:
            # Inisialisasi Nebius client untuk chat
            self.nebius_client = NebiusClient()
            await self.nebius_client.initialize()
            
            # Inisialisasi embedding client dan RAG system
            self.embedding_client = NebiusEmbeddingClient()
            await self.embedding_client.initialize()
            
            self.rag_system = EnhancedRAGSystem(
                embedding_client=self.embedding_client,
                collection_name="sipd_knowledge_base"
            )
            await self.rag_system.initialize()
            
            logger.info("Nebius Chatbot berhasil diinisialisasi")
            
        except Exception as e:
            logger.error(f"Error inisialisasi chatbot: {e}")
            raise
    
    async def process_message(self, chat_message: ChatMessage) -> ChatResponse:
        """Proses pesan dari user dan generate response menggunakan Nebius AI"""
        try:
            start_time = datetime.now()
            
            # Ambil atau buat history conversation
            if chat_message.session_id not in self.conversation_history:
                self.conversation_history[chat_message.session_id] = []
            
            history = self.conversation_history[chat_message.session_id]
            
            # Analisis intent dan sentiment
            intent = await self._classify_intent(chat_message.message)
            sentiment = await self._analyze_sentiment(chat_message.message)
            
            # Ambil konteks dari RAG system
            rag_context = await self._get_rag_context(chat_message.message, intent)
            
            # Generate response menggunakan Nebius AI
            response_data = await self._generate_response(
                message=chat_message.message,
                intent=intent,
                sentiment=sentiment,
                context=rag_context,
                history=history[-5:],  # Ambil 5 pesan terakhir untuk konteks
                user_context=chat_message.context
            )
            
            # Tentukan apakah perlu eskalasi
            should_escalate = self._should_escalate(sentiment, intent, history)
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(intent, response_data.get('response', ''))
            
            # Simpan ke history
            conversation_entry = {
                'timestamp': start_time.isoformat(),
                'user_message': chat_message.message,
                'bot_response': response_data.get('response', ''),
                'intent': intent,
                'sentiment': sentiment,
                'should_escalate': should_escalate
            }
            history.append(conversation_entry)
            
            # Batasi history (maksimal 50 pesan)
            if len(history) > 50:
                history = history[-50:]
            self.conversation_history[chat_message.session_id] = history
            
            # Hitung processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ChatResponse(
                response=response_data.get('response', 'Maaf, terjadi kesalahan dalam memproses pesan Anda.'),
                session_id=chat_message.session_id,
                intent=intent,
                sentiment=sentiment,
                confidence=response_data.get('confidence', 0.0),
                suggestions=suggestions,
                should_escalate=should_escalate,
                metadata={
                    'processing_time': processing_time,
                    'model_used': 'nebius-ai',
                    'rag_context_used': bool(rag_context),
                    'timestamp': start_time.isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return ChatResponse(
                response="Maaf, terjadi kesalahan sistem. Silakan coba lagi atau hubungi admin SIPD.",
                session_id=chat_message.session_id,
                should_escalate=True,
                metadata={'error': str(e)}
            )
    
    async def _classify_intent(self, message: str) -> str:
        """Klasifikasi intent menggunakan Nebius AI"""
        try:
            intent_prompt = f"""
Klasifikasikan intent dari pesan user berikut dalam konteks SIPD:

Pesan: "{message}"

Kembalikan salah satu dari kategori berikut:
- login_issue: Masalah login atau akses
- dpa_issue: Masalah DPA (upload, validasi, persetujuan)
- laporan_issue: Masalah laporan (akses, export, tampilan)
- technical_issue: Masalah teknis sistem
- general_inquiry: Pertanyaan umum
- complaint: Keluhan atau kritik
- praise: Pujian atau feedback positif

Jawab hanya dengan nama kategori:
"""
            
            response = await self.nebius_client.generate_response(
                prompt=intent_prompt,
                max_tokens=50,
                temperature=0.1
            )
            
            # Ekstrak intent dari response
            intent = response.strip().lower()
            valid_intents = ['login_issue', 'dpa_issue', 'laporan_issue', 'technical_issue', 
                           'general_inquiry', 'complaint', 'praise']
            
            return intent if intent in valid_intents else 'general_inquiry'
            
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return 'general_inquiry'
    
    async def _analyze_sentiment(self, message: str) -> str:
        """Analisis sentiment menggunakan Nebius AI"""
        try:
            sentiment_prompt = f"""
Analisis sentiment dari pesan berikut:

Pesan: "{message}"

Kembalikan salah satu dari:
- positive: Sentiment positif
- neutral: Sentiment netral
- negative: Sentiment negatif

Jawab hanya dengan nama sentiment:
"""
            
            response = await self.nebius_client.generate_response(
                prompt=sentiment_prompt,
                max_tokens=20,
                temperature=0.1
            )
            
            sentiment = response.strip().lower()
            valid_sentiments = ['positive', 'neutral', 'negative']
            
            return sentiment if sentiment in valid_sentiments else 'neutral'
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 'neutral'
    
    async def _get_rag_context(self, message: str, intent: str) -> str:
        """Ambil konteks relevan dari RAG system"""
        try:
            if self.rag_system:
                context = await self.rag_system.get_context_for_query(
                    query=message,
                    intent=intent,
                    max_context_length=1000
                )
                return context
            return ""
        except Exception as e:
            logger.error(f"Error getting RAG context: {e}")
            return ""
    
    async def _generate_response(
        self, 
        message: str, 
        intent: str, 
        sentiment: str, 
        context: str, 
        history: List[Dict], 
        user_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate response menggunakan Nebius AI dengan konteks lengkap"""
        try:
            # Pilih system prompt berdasarkan intent
            system_prompt = self.system_prompts.get(intent, self.system_prompts['default'])
            
            # Build conversation context
            conversation_context = ""
            if history:
                conversation_context = "\n\nRiwayat percakapan terakhir:\n"
                for entry in history[-3:]:  # Ambil 3 percakapan terakhir
                    conversation_context += f"User: {entry.get('user_message', '')}\n"
                    conversation_context += f"Bot: {entry.get('bot_response', '')}\n"
            
            # Build RAG context
            rag_context_text = ""
            if context:
                rag_context_text = f"\n\nInformasi relevan dari knowledge base:\n{context}"
            
            # Build user context
            user_context_text = ""
            if user_context:
                user_context_text = f"\n\nKonteks user: {json.dumps(user_context, ensure_ascii=False)}"
            
            # Construct full prompt
            full_prompt = f"""{system_prompt}

Intent yang terdeteksi: {intent}
Sentiment: {sentiment}
{conversation_context}
{rag_context_text}
{user_context_text}

Pesan user saat ini: "{message}"

Berikan response yang sesuai, empati, dan membantu. Jika ini adalah masalah teknis yang kompleks, berikan langkah-langkah troubleshooting yang jelas.

Response:
"""
            
            # Generate response dengan Nebius AI
            response = await self.nebius_client.generate_response(
                prompt=full_prompt,
                max_tokens=500,
                temperature=0.7
            )
            
            return {
                'response': response.strip(),
                'confidence': 0.85  # Placeholder confidence score
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': "Maaf, saya mengalami kesulitan memproses permintaan Anda. Silakan coba lagi.",
                'confidence': 0.0
            }
    
    def _should_escalate(self, sentiment: str, intent: str, history: List[Dict]) -> bool:
        """Tentukan apakah perlu eskalasi ke human agent"""
        # Eskalasi jika sentiment sangat negatif
        if sentiment == 'negative':
            # Cek apakah ada keluhan berulang
            recent_complaints = sum(1 for entry in history[-5:] 
                                  if entry.get('sentiment') == 'negative')
            if recent_complaints >= 2:
                return True
        
        # Eskalasi untuk complaint yang kompleks
        if intent == 'complaint':
            return True
        
        # Eskalasi jika user sudah bertanya hal yang sama berkali-kali
        if len(history) >= 3:
            recent_intents = [entry.get('intent') for entry in history[-3:]]
            if len(set(recent_intents)) == 1 and recent_intents[0] in ['technical_issue', 'login_issue']:
                return True
        
        return False
    
    async def _generate_suggestions(self, intent: str, response: str) -> List[str]:
        """Generate suggestions berdasarkan intent dan response"""
        suggestions_map = {
            'login_issue': [
                "Coba reset password melalui menu 'Lupa Password'",
                "Pastikan browser sudah diupdate ke versi terbaru",
                "Hapus cache dan cookies browser",
                "Hubungi admin SIPD jika masalah berlanjut"
            ],
            'dpa_issue': [
                "Periksa format file DPA (harus .xlsx atau .xls)",
                "Pastikan semua kolom wajib sudah diisi",
                "Cek koneksi internet saat upload",
                "Download template DPA terbaru"
            ],
            'laporan_issue': [
                "Coba refresh halaman laporan",
                "Periksa filter tanggal yang dipilih",
                "Gunakan browser Chrome atau Firefox",
                "Export laporan dalam format PDF jika Excel bermasalah"
            ],
            'technical_issue': [
                "Restart browser dan coba lagi",
                "Periksa koneksi internet",
                "Coba akses dari komputer lain",
                "Laporkan ke tim IT jika masalah persisten"
            ]
        }
        
        return suggestions_map.get(intent, [
            "Apakah ada hal lain yang bisa saya bantu?",
            "Silakan hubungi admin SIPD untuk bantuan lebih lanjut"
        ])
    
    async def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Ambil riwayat percakapan untuk session tertentu"""
        return self.conversation_history.get(session_id, [])
    
    async def clear_conversation_history(self, session_id: str) -> bool:
        """Hapus riwayat percakapan untuk session tertentu"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            return True
        return False
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Ambil status kesehatan chatbot"""
        try:
            nebius_status = await self.nebius_client.health_check() if self.nebius_client else False
            rag_status = self.rag_system is not None
            
            return {
                'status': 'healthy' if nebius_status and rag_status else 'degraded',
                'nebius_connection': nebius_status,
                'rag_system': rag_status,
                'active_sessions': len(self.conversation_history),
                'total_conversations': sum(len(history) for history in self.conversation_history.values()),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global chatbot instance
chatbot = NebiusChatbot()

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Memulai Nebius Chatbot...")
    await chatbot.initialize()
    logger.info("Nebius Chatbot siap digunakan!")
    yield
    # Shutdown
    logger.info("Mematikan Nebius Chatbot...")

# FastAPI app
app = FastAPI(
    title="SIPD Nebius Chatbot",
    description="Chatbot SIPD dengan integrasi Nebius AI untuk response generation yang intelligent",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Halaman chat interface"""
    return """
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
                background: linear-gradient(135deg, #4CAF50, #45a049);
                color: white;
                padding: 20px;
                text-align: center;
                position: relative;
            }
            
            .chat-header h1 {
                font-size: 24px;
                margin-bottom: 5px;
            }
            
            .chat-header p {
                opacity: 0.9;
                font-size: 14px;
            }
            
            .status-indicator {
                position: absolute;
                top: 20px;
                right: 20px;
                width: 12px;
                height: 12px;
                background: #4CAF50;
                border-radius: 50%;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
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
                background: #007bff;
                color: white;
                border-bottom-right-radius: 4px;
            }
            
            .message.bot .message-content {
                background: white;
                color: #333;
                border: 1px solid #e0e0e0;
                border-bottom-left-radius: 4px;
            }
            
            .message-meta {
                font-size: 11px;
                opacity: 0.7;
                margin-top: 4px;
            }
            
            .suggestions {
                margin-top: 10px;
            }
            
            .suggestion-chip {
                display: inline-block;
                background: #e3f2fd;
                color: #1976d2;
                padding: 6px 12px;
                margin: 4px 4px 4px 0;
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
            
            .escalation-notice {
                background: #fff3cd;
                color: #856404;
                padding: 8px 12px;
                border-radius: 8px;
                font-size: 12px;
                margin-top: 8px;
                border-left: 4px solid #ffc107;
            }
            
            .chat-input {
                padding: 20px;
                background: white;
                border-top: 1px solid #e0e0e0;
            }
            
            .input-container {
                display: flex;
                gap: 10px;
                align-items: flex-end;
            }
            
            .message-input {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e0e0e0;
                border-radius: 25px;
                font-size: 14px;
                resize: none;
                max-height: 100px;
                min-height: 44px;
                outline: none;
                transition: border-color 0.2s;
            }
            
            .message-input:focus {
                border-color: #007bff;
            }
            
            .send-button {
                background: #007bff;
                color: white;
                border: none;
                border-radius: 50%;
                width: 44px;
                height: 44px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background-color 0.2s;
            }
            
            .send-button:hover:not(:disabled) {
                background: #0056b3;
            }
            
            .send-button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            
            .typing-indicator {
                display: none;
                padding: 12px 16px;
                background: white;
                border-radius: 18px;
                border-bottom-left-radius: 4px;
                max-width: 70%;
                border: 1px solid #e0e0e0;
            }
            
            .typing-dots {
                display: flex;
                gap: 4px;
            }
            
            .typing-dot {
                width: 8px;
                height: 8px;
                background: #999;
                border-radius: 50%;
                animation: typing 1.4s infinite;
            }
            
            .typing-dot:nth-child(2) {
                animation-delay: 0.2s;
            }
            
            .typing-dot:nth-child(3) {
                animation-delay: 0.4s;
            }
            
            @keyframes typing {
                0%, 60%, 100% {
                    transform: translateY(0);
                }
                30% {
                    transform: translateY(-10px);
                }
            }
            
            .error-message {
                background: #f8d7da;
                color: #721c24;
                padding: 8px 12px;
                border-radius: 8px;
                font-size: 12px;
                margin-top: 8px;
                border-left: 4px solid #dc3545;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <div class="status-indicator"></div>
                <h1>🤖 SIPD Nebius Chatbot</h1>
                <p>Asisten AI untuk Sistem Informasi Pemerintah Daerah</p>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message bot">
                    <div class="message-content">
                        Selamat datang di SIPD Chatbot! 👋<br><br>
                        Saya adalah asisten AI yang terhubung dengan Nebius AI untuk membantu Anda dengan:
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li>Masalah login dan akses sistem</li>
                            <li>Upload dan pengelolaan DPA</li>
                            <li>Pembuatan dan akses laporan</li>
                            <li>Troubleshooting masalah teknis</li>
                            <li>Panduan penggunaan fitur SIPD</li>
                        </ul>
                        Silakan tanyakan apa yang bisa saya bantu! 😊
                    </div>
                </div>
            </div>
            
            <div class="typing-indicator" id="typingIndicator">
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
            
            <div class="chat-input">
                <div class="input-container">
                    <textarea 
                        id="messageInput" 
                        class="message-input" 
                        placeholder="Ketik pesan Anda di sini..."
                        rows="1"
                    ></textarea>
                    <button id="sendButton" class="send-button">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                        </svg>
                    </button>
                </div>
            </div>
        </div>
        
        <script>
            const chatMessages = document.getElementById('chatMessages');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            const typingIndicator = document.getElementById('typingIndicator');
            
            let sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            
            // Auto-resize textarea
            messageInput.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 100) + 'px';
            });
            
            // Send message on Enter (but allow Shift+Enter for new line)
            messageInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            
            sendButton.addEventListener('click', sendMessage);
            
            async function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;
                
                // Disable input
                messageInput.disabled = true;
                sendButton.disabled = true;
                
                // Add user message
                addMessage(message, 'user');
                messageInput.value = '';
                messageInput.style.height = 'auto';
                
                // Show typing indicator
                showTypingIndicator();
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            session_id: sessionId
                        })
                    });
                    
                    const data = await response.json();
                    
                    // Hide typing indicator
                    hideTypingIndicator();
                    
                    if (response.ok) {
                        // Add bot response
                        addMessage(data.response, 'bot', {
                            intent: data.intent,
                            sentiment: data.sentiment,
                            confidence: data.confidence,
                            suggestions: data.suggestions,
                            should_escalate: data.should_escalate,
                            metadata: data.metadata
                        });
                    } else {
                        addMessage('Maaf, terjadi kesalahan. Silakan coba lagi.', 'bot', { error: true });
                    }
                } catch (error) {
                    hideTypingIndicator();
                    addMessage('Maaf, tidak dapat terhubung ke server. Periksa koneksi internet Anda.', 'bot', { error: true });
                }
                
                // Re-enable input
                messageInput.disabled = false;
                sendButton.disabled = false;
                messageInput.focus();
            }
            
            function addMessage(content, sender, metadata = {}) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.innerHTML = content.replace(/\n/g, '<br>');
                
                messageDiv.appendChild(contentDiv);
                
                // Add metadata for bot messages
                if (sender === 'bot' && metadata) {
                    if (metadata.intent || metadata.sentiment) {
                        const metaDiv = document.createElement('div');
                        metaDiv.className = 'message-meta';
                        metaDiv.innerHTML = `Intent: ${metadata.intent || 'N/A'} | Sentiment: ${metadata.sentiment || 'N/A'}`;
                        contentDiv.appendChild(metaDiv);
                    }
                    
                    // Add suggestions
                    if (metadata.suggestions && metadata.suggestions.length > 0) {
                        const suggestionsDiv = document.createElement('div');
                        suggestionsDiv.className = 'suggestions';
                        
                        metadata.suggestions.forEach(suggestion => {
                            const chip = document.createElement('span');
                            chip.className = 'suggestion-chip';
                            chip.textContent = suggestion;
                            chip.onclick = () => {
                                messageInput.value = suggestion;
                                messageInput.focus();
                            };
                            suggestionsDiv.appendChild(chip);
                        });
                        
                        contentDiv.appendChild(suggestionsDiv);
                    }
                    
                    // Add escalation notice
                    if (metadata.should_escalate) {
                        const escalationDiv = document.createElement('div');
                        escalationDiv.className = 'escalation-notice';
                        escalationDiv.innerHTML = '⚠️ Masalah ini akan diteruskan ke admin SIPD untuk penanganan lebih lanjut.';
                        contentDiv.appendChild(escalationDiv);
                    }
                    
                    // Add error notice
                    if (metadata.error) {
                        const errorDiv = document.createElement('div');
                        errorDiv.className = 'error-message';
                        errorDiv.innerHTML = '❌ Terjadi kesalahan sistem';
                        contentDiv.appendChild(errorDiv);
                    }
                }
                
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function showTypingIndicator() {
                typingIndicator.style.display = 'block';
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function hideTypingIndicator() {
                typingIndicator.style.display = 'none';
            }
            
            // Focus on input when page loads
            messageInput.focus();
        </script>
    </body>
    </html>
    """

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage, request: Request):
    """Endpoint utama untuk chat dengan Nebius AI"""
    try:
        # Process message dengan chatbot
        response = await chatbot.process_message(chat_message)
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = await chatbot.get_health_status()
    return health_status

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Ambil riwayat chat untuk session tertentu"""
    try:
        history = await chatbot.get_conversation_history(session_id)
        return {
            "session_id": session_id,
            "history": history,
            "total_messages": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Hapus riwayat chat untuk session tertentu"""
    try:
        success = await chatbot.clear_conversation_history(session_id)
        return {
            "session_id": session_id,
            "cleared": success
        }
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/stats")
async def get_chatbot_stats():
    """Statistik penggunaan chatbot"""
    try:
        total_sessions = len(chatbot.conversation_history)
        total_messages = sum(len(history) for history in chatbot.conversation_history.values())
        
        # Analisis intent dan sentiment
        intent_counts = {}
        sentiment_counts = {}
        
        for history in chatbot.conversation_history.values():
            for entry in history:
                intent = entry.get('intent', 'unknown')
                sentiment = entry.get('sentiment', 'unknown')
                
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "avg_messages_per_session": total_messages / max(total_sessions, 1),
            "intent_distribution": intent_counts,
            "sentiment_distribution": sentiment_counts,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(
        "nebius_chatbot:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )