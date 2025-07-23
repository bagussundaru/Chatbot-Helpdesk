# Simplified Enhanced SIPD AI Chatbot Application
# A simplified version that doesn't require sentence-transformers

import os
import json
import asyncio
import uvicorn
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
from contextlib import asynccontextmanager
import random

# Import configuration
from config import settings

# Models for API
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="Pesan dari user")
    session_id: Optional[str] = Field(None, description="ID sesi chat")
    user_id: Optional[str] = Field(None, description="ID user (opsional)")
    context: Optional[Dict[str, Any]] = Field(None, description="Konteks tambahan")
    language: Optional[str] = Field(None, description="Bahasa yang digunakan user (opsional)")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Response dari chatbot")
    session_id: str = Field(..., description="ID sesi chat")
    detected_language: Optional[str] = Field(None, description="Bahasa yang terdeteksi")
    intent: Optional[str] = Field(None, description="Intent yang terdeteksi")
    sentiment: Optional[str] = Field(None, description="Sentiment yang terdeteksi")
    confidence: Optional[float] = Field(None, description="Confidence score")
    suggestions: List[str] = Field(default_factory=list, description="Saran tindak lanjut")
    should_escalate: bool = Field(False, description="Apakah perlu eskalasi ke human agent")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata tambahan")

class SimplifiedEnhancedChatbot:
    """Simplified Enhanced SIPD Chatbot with multilingual support"""
    
    def __init__(self):
        self.conversation_history = {}
        
        # System prompts for different languages
        self.system_prompts = {
            "id": """Anda adalah asisten AI untuk Sistem Informasi Pemerintah Daerah (SIPD). 
Berikan jawaban yang akurat, ramah, dan profesional dalam Bahasa Indonesia.""",
            "en": """You are an AI assistant for the Regional Government Information System (SIPD).
Provide accurate, friendly, and professional answers in English.""",
            "jv": """Panjenengan minangka asisten AI kanggo Sistem Informasi Pemerintah Daerah (SIPD).
Nyaosaken wangsulan ingkang akurat, ramah, lan profesional ing basa Jawa.""",
            "su": """Anjeun mangrupa asisten AI pikeun Sistem Informasi Pamaréntah Daérah (SIPD).
Masihan jawaban anu akurat, ramah, sareng profésional dina basa Sunda.""",
            "ms": """Anda adalah pembantu AI untuk Sistem Maklumat Kerajaan Daerah (SIPD).
Berikan jawapan yang tepat, mesra, dan profesional dalam Bahasa Melayu.""",
            "default": """Anda adalah asisten AI untuk Sistem Informasi Pemerintah Daerah (SIPD).
Berikan jawaban yang akurat, ramah, dan profesional."""
        }
        
        # Sample responses for different intents and languages
        self.sample_responses = {
            "id": {
                "login_issue": "Hai! Saya mengerti Anda mengalami masalah login SIPD. Jangan khawatir, ini sering terjadi dan biasanya bisa diatasi dengan mudah. Mari kita coba beberapa langkah berikut ya:\n\n1. Pastikan username dan password Anda sudah benar (perhatikan huruf besar/kecil)\n2. Coba bersihkan cache browser Anda (biasanya dengan Ctrl+Shift+Delete)\n3. Jika masih bermasalah, coba pakai browser lain seperti Chrome atau Firefox\n4. Masih belum bisa juga? Saya bisa bantu hubungkan Anda dengan admin SIPD\n\nBagaimana, sudah dicoba langkah-langkah di atas?",
                "dpa_issue": "Halo! Saya paham masalah DPA bisa cukup membingungkan. Tenang saja, kita akan cari solusinya bersama. Beberapa hal yang perlu diperhatikan:\n\n1. Pastikan semua field yang wajib diisi sudah terisi dengan lengkap\n2. Periksa format data yang Anda masukkan (terutama angka dan tanggal)\n3. Pastikan koneksi internet Anda stabil saat mengisi DPA\n4. Kadang refresh halaman bisa membantu mengatasi masalah\n\nApakah ada pesan error spesifik yang muncul? Itu bisa membantu saya mendiagnosis masalahnya lebih tepat.",
                "laporan_issue": "Hai! Masalah dengan laporan ya? Saya mengerti ini bisa mengganggu pekerjaan Anda. Mari kita selesaikan bersama dengan langkah-langkah berikut:\n\n1. Cek format data yang Anda gunakan dalam laporan\n2. Pastikan periode laporan yang Anda pilih sudah tepat\n3. Jika laporan terlalu besar, coba export dengan data yang lebih sedikit dulu\n4. Jika masih bermasalah, saya bisa bantu menghubungkan Anda dengan tim teknis kami\n\nBoleh ceritakan lebih detail tentang laporan apa yang sedang Anda coba buat?",
                "general_question": "Halo! Senang bisa berbincang dengan Anda hari ini. Saya Asisten SIPD yang siap membantu dengan pertanyaan apapun seputar sistem. Saya akan berusaha memberikan jawaban sebaik mungkin dengan bahasa yang mudah dipahami. Jadi, ada yang bisa saya bantu hari ini? Ceritakan saja, saya siap mendengarkan!",
                "other": "Halo! Terima kasih sudah menghubungi Asisten SIPD. Saya di sini untuk membantu Anda dengan segala pertanyaan atau masalah seputar SIPD. Ceritakan saja apa yang sedang Anda alami, dan saya akan berusaha memberikan solusi terbaik. Jangan sungkan ya, anggap saja saya teman Anda dalam menggunakan SIPD!"
            },
            "en": {
                "login_issue": "Hi there! I understand you're having trouble logging into SIPD. Don't worry, this is common and usually easy to fix. Let's try these steps together:\n\n1. Make sure your username and password are correct (remember they're case-sensitive)\n2. Try clearing your browser cache (usually with Ctrl+Shift+Delete)\n3. If it's still not working, try using a different browser like Chrome or Firefox\n4. Still having issues? I can help connect you with a SIPD admin\n\nHave you tried any of these steps already?",
                "dpa_issue": "Hello! I understand DPA issues can be quite confusing. Don't worry, we'll find a solution together. Here are some things to check:\n\n1. Make sure all mandatory fields are filled in completely\n2. Check the format of the data you're entering (especially numbers and dates)\n3. Ensure your internet connection is stable when filling out the DPA\n4. Sometimes refreshing the page can help resolve issues\n\nIs there a specific error message appearing? That could help me diagnose the problem more accurately.",
                "laporan_issue": "Hi there! Having trouble with reports? I understand this can disrupt your work. Let's solve this together with these steps:\n\n1. Check the data format you're using in the report\n2. Make sure the reporting period you've selected is correct\n3. If the report is too large, try exporting with less data first\n4. If you're still having issues, I can help connect you with our technical team\n\nCould you tell me more about what kind of report you're trying to create?",
                "general_question": "Hello! It's great to chat with you today. I'm your SIPD Assistant, ready to help with any questions about the system. I'll do my best to provide clear, easy-to-understand answers. So, what can I help you with today? I'm all ears!",
                "other": "Hello! Thanks for reaching out to the SIPD Assistant. I'm here to help you with any questions or issues related to SIPD. Just let me know what you're experiencing, and I'll do my best to provide the best solution. Don't hesitate - think of me as your friend in navigating SIPD!"
            }
        }
        
        # Add default responses for other languages
        for lang in ["jv", "su", "ms"]:
            self.sample_responses[lang] = self.sample_responses["id"]
    
    async def initialize(self):
        """Initialize all components"""
        try:
            print("Simplified Enhanced SIPD Chatbot initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing chatbot: {e}")
            return False
    
    async def process_message(self, chat_message: ChatMessage) -> ChatResponse:
        """Process incoming message and generate response"""
        try:
            start_time = datetime.now()
            
            # Generate session ID if not provided
            session_id = chat_message.session_id or str(uuid.uuid4())
            
            # Initialize conversation history if needed
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            
            history = self.conversation_history[session_id]
            
            # Detect language (simplified)
            detected_language = chat_message.language or self._detect_language(chat_message.message)
            
            # Classify intent (simplified)
            intent = self._classify_intent(chat_message.message)
            
            # Analyze sentiment (simplified)
            sentiment = self._analyze_sentiment(chat_message.message)
            
            # Generate response (simplified)
            response_text = self._generate_response(chat_message.message, detected_language, intent)
            
            # Generate suggestions based on intent and language
            suggestions = self._generate_suggestions(intent, detected_language)
            
            # Determine if escalation is needed (simplified)
            should_escalate = self._should_escalate(chat_message.message, response_text, sentiment, history)
            
            # Update conversation history
            history.append({
                "timestamp": datetime.now().isoformat(),
                "user_message": chat_message.message,
                "bot_response": response_text,
                "detected_language": detected_language,
                "intent": intent,
                "sentiment": sentiment,
                "should_escalate": should_escalate
            })
            
            # Limit history size
            if len(history) > 50:
                history = history[-50:]
            self.conversation_history[session_id] = history
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ChatResponse(
                response=response_text,
                session_id=session_id,
                detected_language=detected_language,
                intent=intent,
                sentiment=sentiment,
                confidence=0.9,  # Placeholder confidence score
                suggestions=suggestions,
                should_escalate=should_escalate,
                metadata={
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            print(f"Error processing message: {e}")
            return ChatResponse(
                response="Maaf, terjadi kesalahan sistem. Silakan coba lagi atau hubungi admin SIPD.",
                session_id=session_id,
                should_escalate=True,
                metadata={"error": str(e)}
            )
    
    def _detect_language(self, text: str) -> str:
        """Simplified language detection"""
        # Simple keyword-based language detection
        text_lower = text.lower()
        
        # English keywords
        english_keywords = ["the", "is", "are", "what", "how", "when", "where", "why", "who", "which"]
        if any(keyword in text_lower.split() for keyword in english_keywords):
            return "en"
        
        # Default to Indonesian
        return "id"
    
    def _classify_intent(self, message: str) -> str:
        """Simplified intent classification"""
        message_lower = message.lower()
        
        # Login issues
        if any(word in message_lower for word in ["login", "masuk", "akses", "password", "username", "user", "pass"]):
            return "login_issue"
        
        # DPA issues
        if any(word in message_lower for word in ["dpa", "anggaran", "budget", "dana", "input", "entry"]):
            return "dpa_issue"
        
        # Report issues
        if any(word in message_lower for word in ["laporan", "report", "export", "excel", "cetak", "print"]):
            return "laporan_issue"
        
        # General questions
        if any(word in message_lower for word in ["apa", "bagaimana", "what", "how", "kenapa", "why", "kapan", "when"]):
            return "general_question"
        
        # Default to other
        return "other"
    
    def _analyze_sentiment(self, message: str) -> str:
        """Simplified sentiment analysis with expanded keywords for better emotion detection"""
        message_lower = message.lower()
        
        # Positive keywords - expanded to include more expressions of satisfaction and gratitude
        positive_keywords = [
            # Indonesian positive words
            "bagus", "baik", "hebat", "keren", "mantap", "oke", "ok", "berhasil", "sukses", "senang", "puas", 
            "terima kasih", "makasih", "thx", "trims", "luar biasa", "membantu", "berguna", "bermanfaat", "sip", "jos",
            # English positive words
            "good", "great", "excellent", "awesome", "thanks", "thank you", "helpful", "useful", "success", "successful",
            "appreciate", "nice", "wonderful", "fantastic", "perfect", "solved", "working", "works", "happy", "glad"
        ]
        
        # Check for positive phrases (not just single words)
        positive_phrases = [
            "sangat membantu", "sangat berguna", "sangat baik", "very helpful", "very useful", "very good",
            "terima kasih banyak", "thank you so much", "masalah teratasi", "problem solved"
        ]
        
        # Negative keywords - expanded to include frustration and dissatisfaction
        negative_keywords = [
            # Indonesian negative words
            "error", "gagal", "tidak bisa", "tidak berhasil", "masalah", "problem", "rusak", "bug", "crash", "lambat", 
            "lama", "susah", "sulit", "rumit", "bingung", "kecewa", "marah", "kesal", "jengkel", "buruk", "jelek",
            # English negative words
            "fail", "failed", "can't", "cannot", "issue", "broken", "slow", "difficult", "confusing", "confused",
            "disappointed", "angry", "upset", "bad", "terrible", "horrible", "doesn't work", "not working", "stuck"
        ]
        
        # Check for negative phrases
        negative_phrases = [
            "tidak membantu", "tidak berguna", "sangat buruk", "not helpful", "not useful", "very bad",
            "masih error", "still error", "masih bermasalah", "still problematic", "semakin parah", "getting worse"
        ]
        
        # Check for positive sentiment
        if any(word in message_lower.split() for word in positive_keywords) or \
           any(phrase in message_lower for phrase in positive_phrases):
            return "positive"
        
        # Check for negative sentiment
        if any(word in message_lower.split() for word in negative_keywords) or \
           any(phrase in message_lower for phrase in negative_phrases):
            return "negative"
        
        # Default to neutral
        return "neutral"
    
    def _generate_response(self, message: str, language: str, intent: str) -> str:
        """Simplified response generation"""
        # Get language-specific responses or default to Indonesian
        lang_responses = self.sample_responses.get(language, self.sample_responses["id"])
        
        # Get intent-specific response or default to other
        response = lang_responses.get(intent, lang_responses["other"])
        
        return response
    
    def _generate_suggestions(self, intent: str, language: str) -> List[str]:
        """Generate contextual suggestions based on intent and language"""
        # Suggestions for Indonesian - more conversational and helpful
        suggestions_id = {
            "login_issue": [
                "Saya lupa password",
                "Koneksi internet saya tidak stabil",
                "Bagaimana cara menghubungi admin?"
            ],
            "dpa_issue": [
                "Format data yang benar seperti apa?",
                "Field apa saja yang wajib diisi?",
                "DPA saya tidak bisa disimpan"
            ],
            "laporan_issue": [
                "Periode laporan tidak muncul",
                "Cara export ke Excel",
                "Laporan tidak sesuai data"
            ],
            "general_question": [
                "Cara menggunakan SIPD",
                "Fitur baru di SIPD",
                "Jadwal maintenance SIPD"
            ],
            "other": [
                "Saya butuh bantuan login",
                "Ada masalah dengan DPA",
                "Laporan tidak bisa diakses"
            ]
        }
        
        # Suggestions for English - more conversational and helpful
        suggestions_en = {
            "login_issue": [
                "I forgot my password",
                "My internet connection is unstable",
                "How do I contact the admin?"
            ],
            "dpa_issue": [
                "What's the correct data format?",
                "Which fields are mandatory?",
                "My DPA can't be saved"
            ],
            "laporan_issue": [
                "Report period doesn't appear",
                "How to export to Excel",
                "Report doesn't match my data"
            ],
            "general_question": [
                "How to use SIPD",
                "New features in SIPD",
                "SIPD maintenance schedule"
            ],
            "other": [
                "I need help with login",
                "I have an issue with DPA",
                "Can't access my reports"
            ]
        }
        
        # Select suggestions based on language
        if language == "en":
            return suggestions_en.get(intent, suggestions_en["other"])
        else:
            return suggestions_id.get(intent, suggestions_id["other"])
    
    def _should_escalate(self, message: str, response: str, sentiment: str, history: List[Dict]) -> bool:
        """Enhanced escalation logic with better detection of user frustration"""
        message_lower = message.lower()
        
        # Escalate if user explicitly asks for human assistance
        human_request_phrases = [
            # Indonesian phrases
            "bicara dengan manusia", "bicara dengan admin", "hubungi admin", "operator", "customer service",
            "cs", "layanan pelanggan", "bantuan manusia", "tidak mau bot", "butuh bantuan langsung",
            # English phrases
            "speak to human", "talk to agent", "human agent", "real person", "customer service",
            "not a bot", "need human", "human assistance", "human support", "live agent"
        ]
        
        if any(phrase in message_lower for phrase in human_request_phrases) or \
           any(word in message_lower.split() for word in ["human", "agent", "manusia", "operator", "admin"]):
            return True
        
        # Escalate if sentiment is negative in consecutive messages
        if sentiment == "negative":
            # Check if this is the second consecutive negative sentiment
            if len(history) >= 1 and history[-1].get('sentiment') == "negative":
                return True
        
        # Escalate if message contains urgent or frustrated keywords
        urgent_keywords = [
            # Indonesian urgency words
            "urgent", "darurat", "segera", "penting", "kritis", "gawat", "mendesak", "secepatnya",
            "tidak sabar", "frustrasi", "kecewa", "marah", "kesal", "jengkel", "tidak membantu",
            # English urgency words
            "emergency", "urgent", "critical", "important", "asap", "immediately", "frustrated",
            "annoyed", "upset", "angry", "unhelpful", "useless", "waste of time"
        ]
        
        if any(word in message_lower.split() for word in urgent_keywords):
            return True
        
        # Escalate if message contains multiple question marks or exclamation points (signs of frustration)
        if message.count('?') >= 3 or message.count('!') >= 2:
            return True
        
        # Escalate if message is very short after several exchanges (might indicate frustration)
        if len(history) >= 3 and len(message.strip()) <= 5:
            return True
        
        return False
    
    async def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        return self.conversation_history.get(session_id, [])
    
    async def clear_conversation_history(self, session_id: str) -> bool:
        """Clear conversation history for a session"""
        if session_id in self.conversation_history:
            self.conversation_history[session_id] = []
            return True
        return False

# Global chatbot instance
chatbot = SimplifiedEnhancedChatbot()

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Simplified Enhanced SIPD Chatbot...")
    await chatbot.initialize()
    print("Simplified Enhanced SIPD Chatbot ready!")
    yield
    # Shutdown
    print("Shutting down Simplified Enhanced SIPD Chatbot...")

# FastAPI app
app = FastAPI(
    title="Simplified Enhanced SIPD Chatbot",
    description="Chatbot SIPD dengan arsitektur yang disederhanakan dan dukungan multilingual",
    version="2.0.0",
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
@app.get("/logo/{filename}")
async def get_logo(filename: str):
    """Endpoint untuk menyajikan file logo"""
    return FileResponse(f"logo/{filename}")

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Halaman chat interface"""
    html_content = r'''<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simplified Enhanced SIPD Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
            
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            transition: background 0.5s ease;
            animation: gradientAnimation 15s ease infinite;
        }
            
        @keyframes gradientAnimation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
            animation: fadeIn 0.7s ease-in-out;
            transition: all 0.3s ease;
        }
            
        .chat-container:hover {
            box-shadow: 0 20px 40px rgba(0,0,0,0.25);
            transform: translateY(-5px);
        }
        
        .chat-header {
            background: linear-gradient(to right, #3498db, #2980b9);
            color: white;
            padding: 25px 20px;
            text-align: center;
            position: relative;
            border-bottom: 1px solid #e0e0e0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
            
        .chat-header h1 {
            font-size: 24px;
            margin-bottom: 5px;
            animation: fadeIn 0.5s ease-in-out;
        }
        
        .chat-header p {
            font-size: 14px;
            opacity: 0.9;
            animation: fadeIn 0.7s ease-in-out;
        }
        
        .language-selector {
            position: absolute;
            right: 20px;
            top: 20px;
            animation: fadeIn 0.5s ease-in-out;
        }
            
        .language-selector select {
            padding: 8px 15px;
            border-radius: 20px;
            border: none;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            /* backdrop-filter: blur(5px); */
        }
        
        .language-selector select:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }
        
        .language-selector select:focus {
            outline: none;
            box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.3);
        }
            
        .chat-messages {
            flex: 1;
            padding: 25px;
            overflow-y: auto;
            background: #f8f9fa;
            background-image: radial-gradient(#e3e3e3 1px, transparent 1px);
            background-size: 20px 20px;
            transition: all 0.3s ease;
        }
        
        .chat-messages::-webkit-scrollbar {
            width: 8px;
        }
        
        .chat-messages::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        .chat-messages::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 10px;
            transition: all 0.3s;
        }
            
        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
        
        .message {
            margin: 10px 0;
            padding: 14px 18px;
            border-radius: 18px;
            max-width: 70%;
            word-wrap: break-word;
            position: relative;
            line-height: 1.5;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            animation: fadeIn 0.3s ease-in-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
            
        .user-message {
            background: linear-gradient(to right, #3498db, #2980b9);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 5px;
            text-align: right;
        }
        
        .bot-message {
            background: linear-gradient(to right, #f5f7fa, #e9ecef);
            color: #333;
            margin-right: auto;
            white-space: pre-line;
            border-bottom-left-radius: 5px;
            position: relative;
        }
        

            
        .suggestions {
            display: flex;
            flex-wrap: wrap;
            margin-top: 15px;
            justify-content: flex-start;
            animation: fadeIn 0.5s ease-in-out;
            gap: 8px;
        }
        
        .suggestion-chip {
            background: linear-gradient(to right, #3498db, #2980b9);
            color: white;
            padding: 10px 18px;
            margin: 4px;
            border-radius: 20px;
            font-size: 14px;
            cursor: pointer;
            border: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .suggestion-chip:hover {
            background: linear-gradient(to right, #2980b9, #3498db);
            transform: translateY(-2px) scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
            
        .chat-input {
            padding: 20px;
            background: white;
            border-top: 1px solid #dee2e6;
            display: flex;
            align-items: center;
            box-shadow: 0 -5px 10px rgba(0,0,0,0.03);
        }
        
        .chat-input input {
            flex: 1;
            padding: 14px 18px;
            border: 2px solid #dee2e6;
            border-radius: 25px;
            outline: none;
            font-size: 16px;
            transition: all 0.3s;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
        }
            
        .chat-input input:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
        }
        
        .chat-input button {
            padding: 14px 28px;
            background: linear-gradient(to right, #3498db, #2980b9);
            color: white;
            border: none;
            border-radius: 25px;
            margin-left: 10px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .chat-input button:hover {
            background: linear-gradient(to right, #2980b9, #3498db);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
            
        .typing-indicator {
            background: linear-gradient(to right, #f5f7fa, #e9ecef);
            color: #333;
            margin-right: auto;
            padding: 14px 18px;
            border-radius: 18px;
            border-bottom-left-radius: 5px;
            display: none;
            position: relative;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            animation: fadeIn 0.3s ease-in-out;
        }
        
        .typing-indicator::before {
            content: '🤖';
            position: absolute;
            left: -25px;
            top: 10px;
            font-size: 16px;
        }
        
        .typing-indicator::after {
            content: "";
            display: inline-block;
            width: 30px;
            text-align: left;
            animation: typing 1.5s infinite;
        }
        
        @keyframes typing {
            0% { content: "."; }
            33% { content: ".."; }
            66% { content: "..."; }
        }
            
        .escalation-notice {
            background: linear-gradient(to right, #ff9966, #ff5e62);
            color: white;
            padding: 15px;
            text-align: center;
            font-weight: bold;
            margin-top: 15px;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            animation: fadeIn 0.5s ease-in-out;
            line-height: 1.5;
        }
        
        .metadata {
            font-size: 12px;
            color: #6c757d;
            margin-top: 5px;
            text-align: right;
        }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <img src="/logo/SIPD.svg" alt="Logo SIPD" style="height: 40px; margin-bottom: 10px;">
                <h1>Asisten Helpdesk SIPD</h1>
                <div class="language-selector">
                    <select id="languageSelect" onchange="changeLanguage()">
                        <option value="id">Bahasa Indonesia</option>
                        <option value="en">English</option>
                        <option value="jv">Basa Jawa</option>
                        <option value="su">Basa Sunda</option>
                        <option value="ms">Bahasa Melayu</option>
                    </select>
                </div>
            </div>
            <div class="chat-messages" id="chatMessages">
                <div class="message bot-message">
                    <img src="/logo/SIPD.svg" alt="Logo SIPD" style="height: 24px; margin-right: 8px; vertical-align: middle;">
                    Halo! Terima kasih sudah menghubungi Asisten Helpdesk SIPD
                    
                    Saya di sini untuk membantu Anda dengan segala pertanyaan atau masalah seputar SIPD. Anggap saja saya teman Anda dalam menggunakan sistem ini!
                    
                    Beberapa hal yang sering ditanyakan:
                    • 💻 Masalah login
                    • 📊 Masalah DPA/anggaran
                    • 📝 Masalah laporan
                    
                    Ada yang bisa saya bantu hari ini?
                </div>
            </div>
            <div class="chat-input">
                <input type="text" id="messageInput" placeholder="Ketik pesan Anda di sini..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()">Kirim</button>
                <span id="typingIndicator" class="typing-indicator" style="display: none;">Mengetik...</span>
            </div>
        </div>

        </div>
        <script>
            // Global variables
            let sessionId = null;
            let currentLanguage = 'id';
            
            // Initialize when DOM is loaded
            document.addEventListener('DOMContentLoaded', function() {
                sessionId = generateSessionId();
            });
            
            function generateSessionId() {
                return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
            
            function changeLanguage() {
                const select = document.getElementById('languageSelect');
                currentLanguage = select.value;
                
                // Update placeholder text based on language
                const messageInput = document.getElementById('messageInput');
                const sendButton = document.querySelector('.chat-input button');
                
                const placeholders = {
                    "id": "Ketik pesan Anda di sini...",
                    "en": "Type your message here...",
                    "jv": "Ketik pesen panjenengan ing kene...",
                    "su": "Ketik pesen anjeun di dieu...",
                    "ms": "Taip mesej anda di sini..."
                };
                
                const buttonTexts = {
                    "id": "Kirim",
                    "en": "Send",
                    "jv": "Kirim",
                    "su": "Kirim",
                    "ms": "Hantar"
                };
                
                messageInput.placeholder = placeholders[currentLanguage] || placeholders['id'];
                sendButton.textContent = buttonTexts[currentLanguage] || buttonTexts['id'];
                
                // Add welcome message in selected language
                const welcomeMessages = {
                    "id": "Bahasa telah diubah ke Bahasa Indonesia. Selamat datang di Asisten SIPD 👋\n\nSaya di sini untuk membantu Anda dengan segala pertanyaan atau masalah seputar SIPD. Anggap saja saya teman Anda dalam menggunakan sistem ini!",
                    "en": "Language has been changed to English. Welcome to the SIPD Assistant 👋\n\nI'm here to help you with any questions or issues related to SIPD. Think of me as your friend in navigating this system!",
                    "jv": "Basa sampun diganti dados Basa Jawa. Sugeng rawuh ing Asisten SIPD 👋\n\nKula ing ngriki kangge mbiyantu panjenengan kaliyan sedaya pitakenan utawi masalah ingkang gegayutan kaliyan SIPD. Anggep mawon kula kanca panjenengan wonten ing sistem punika!",
                    "su": "Basa geus dirobah jadi Basa Sunda. Wilujeng sumping di Asisten SIPD 👋\n\nAbdi di dieu pikeun mantuan anjeun jeung sagala patarosan atawa masalah ngeunaan SIPD. Anggap wae abdi babaturan anjeun dina ngagunakeun sistem ieu!",
                    "ms": "Bahasa telah ditukar kepada Bahasa Melayu. Selamat datang ke Pembantu SIPD 👋\n\nSaya di sini untuk membantu anda dengan sebarang soalan atau masalah berkaitan SIPD. Anggaplah saya sebagai rakan anda dalam menggunakan sistem ini!"
                };
                
                addMessage(welcomeMessages[currentLanguage] || welcomeMessages['id'], 'bot');
            }
            
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                
                if (!message) return;
                
                // Add user message to chat
                addMessage(message, 'user');
                input.value = '';
                
                // Show typing indicator
                document.getElementById('typingIndicator').style.display = 'inline';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            session_id: sessionId,
                            language: currentLanguage
                        })
                    });
                    
                    const data = await response.json();
                    
                    // Hide typing indicator
                    document.getElementById('typingIndicator').style.display = 'none';
                    
                    // Add bot response
                    addMessage(data.response, 'bot', data.metadata);
                    
                    // Add escalation notice if needed
                    if (data.should_escalate) {
                        addEscalationNotice(data.detected_language);
                    }
                    
                    // Add suggestions if available
                    if (data.suggestions && data.suggestions.length > 0) {
                        addSuggestions(data.suggestions);
                    }
                    
                } catch (error) {
                    document.getElementById('typingIndicator').style.display = 'none';
                    addMessage('Maaf, terjadi kesalahan. Silakan coba lagi.', 'bot');
                    console.error('Error:', error);
                }
            }
            
            function addMessage(message, sender, metadata = null) {
                const chatMessages = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                messageDiv.textContent = message;
                
                // Add metadata if available
                if (metadata && sender === 'bot') {
                    const metadataDiv = document.createElement('div');
                    metadataDiv.className = 'metadata';
                    
                    // Format metadata
                    let metadataText = '';
                    if (metadata.processing_time) {
                        metadataText += `Waktu proses: ${metadata.processing_time.toFixed(2)}s`;
                    }
                    
                    metadataDiv.textContent = metadataText;
                    messageDiv.appendChild(metadataDiv);
                }
                
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function addSuggestions(suggestions) {
                const chatMessages = document.getElementById('chatMessages');
                const suggestionsDiv = document.createElement('div');
                suggestionsDiv.className = 'suggestions';
                
                suggestions.forEach(suggestion => {
                    const chip = document.createElement('span');
                    chip.className = 'suggestion-chip';
                    chip.textContent = suggestion;
                    chip.onclick = () => {
                        document.getElementById('messageInput').value = suggestion;
                        sendMessage();
                    };
                    suggestionsDiv.appendChild(chip);
                });
                
                chatMessages.appendChild(suggestionsDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function addEscalationNotice(language) {
                const chatMessages = document.getElementById('chatMessages');
                const noticeDiv = document.createElement('div');
                noticeDiv.className = 'escalation-notice';
                
                const escalationMessages = {
                    "id": "Sepertinya Anda membutuhkan bantuan lebih lanjut. Saya akan menghubungkan Anda dengan staf helpdesk SIPD kami. Mohon tunggu sebentar ya, mereka akan segera membantu Anda. 👨‍💼👩‍💼",
                    "en": "It seems you need further assistance. I will connect you with our SIPD helpdesk staff. Please wait a moment, they will be with you shortly to help solve your issue. 👨‍💼👩‍💼",
                    "jv": "Kados-kados panjenengan mbetahaken bantuan langkung tebih. Kula badhe nyambungaken panjenengan kaliyan staf helpdesk SIPD. Mangga nengga sekedhap, piyambakipun badhe enggal mbiyantu panjenengan. 👨‍💼👩‍💼",
                    "su": "Sigana anjeun butuh bantuan nu leuwih jauh. Abdi bade nyambungkeun anjeun ka staf helpdesk SIPD. Mangga antosan sakedap, aranjeunna bade enggal ngabantuan anjeun. 👨‍💼👩‍💼",
                    "ms": "Nampaknya anda memerlukan bantuan lanjut. Saya akan menghubungkan anda dengan staf helpdesk SIPD kami. Sila tunggu sebentar, mereka akan segera membantu anda. 👨‍💼👩‍💼"
                };
                
                noticeDiv.textContent = escalationMessages[language] || escalationMessages['id'];
                
                chatMessages.appendChild(noticeDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        </script>
    </body>
    </html>
    '''

    return html_content

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage, background_tasks: BackgroundTasks):
    """Main chat endpoint"""
    response = await chatbot.process_message(message)
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(chatbot.conversation_history)
    }

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get conversation history for a session"""
    history = await chatbot.get_conversation_history(session_id)
    return {"session_id": session_id, "history": history}

@app.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear conversation history for a session"""
    success = await chatbot.clear_conversation_history(session_id)
    if success:
        return {"status": "success", "message": f"History for session {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

@app.get("/languages")
async def get_supported_languages():
    """Get supported languages"""
    return {"languages": ["id", "en", "jv", "su", "ms"]}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simplified Enhanced SIPD Chatbot...")
    print("🌐 Access the chatbot at: http://localhost:8000")
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)
    
    uvicorn.run(
        "simplified_enhanced_chatbot:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )