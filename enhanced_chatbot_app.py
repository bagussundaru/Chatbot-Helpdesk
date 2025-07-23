# Enhanced SIPD AI Chatbot Application
# Aplikasi utama untuk SIPD AI Chatbot dengan arsitektur canggih

import os
import json
import asyncio
import uvicorn
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
from datetime import datetime
import uuid
from contextlib import asynccontextmanager

# Import konfigurasi dan komponen
from config import settings
from meta_llm_client import MetaLLMClient
from personalized_knowledge_embeddings import PersonalizedKnowledgeEmbeddings
from secure_api_layer import SecureAPILayer
from language_detector import LanguageDetector

# Setup logging
logger.add("logs/enhanced_chatbot_app.log", rotation="10 MB", level="INFO")

# Models untuk API
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

class EnhancedSIPDChatbot:
    """Enhanced SIPD Chatbot with multilingual support and advanced architecture"""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.llm_client = MetaLLMClient()
        self.knowledge_embeddings = PersonalizedKnowledgeEmbeddings()
        self.secure_api = SecureAPILayer()
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
        
        # Intent classification prompts
        self.intent_classification_prompt = """Klasifikasikan intent dari pesan berikut ke dalam salah satu kategori: login_issue, dpa_issue, laporan_issue, general_question, atau other.

Pesan: {message}

Intent:"""
        
        # Sentiment analysis prompts
        self.sentiment_analysis_prompt = """Analisis sentiment dari pesan berikut dan kategorikan sebagai positive, neutral, atau negative.

Pesan: {message}

Sentiment:"""
    
    async def initialize(self):
        """Initialize all components"""
        try:
            await self.language_detector.initialize()
            await self.llm_client.initialize()
            await self.knowledge_embeddings.initialize()
            
            # Initialize knowledge base if needed
            knowledge_csv_path = os.path.join(os.getcwd(), "data", "sipd_knowledge_base.csv")
            if os.path.exists(knowledge_csv_path):
                logger.info(f"Initializing knowledge base from {knowledge_csv_path}")
                await self.knowledge_embeddings.initialize_from_csv(knowledge_csv_path)
            
            logger.info("Enhanced SIPD Chatbot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
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
            
            # Mask sensitive data
            masked_message = self.secure_api.mask_sensitive_data(chat_message.message)
            
            # Detect language
            detected_language = chat_message.language or await self.language_detector.detect_language(masked_message)
            
            # Classify intent
            intent = await self._classify_intent(masked_message)
            
            # Analyze sentiment
            sentiment = await self._analyze_sentiment(masked_message)
            
            # Get relevant context from knowledge base
            similar_docs = await self.knowledge_embeddings.search_similar(masked_message)
            context = await self.knowledge_embeddings.get_context(masked_message) if similar_docs else ""
            
            # Generate response
            response_data = await self._generate_response(
                message=masked_message,
                language=detected_language,
                intent=intent,
                sentiment=sentiment,
                context=context,
                history=history
            )
            
            # Generate suggestions based on intent and language
            suggestions = self._generate_suggestions(intent, detected_language)
            
            # Determine if escalation is needed
            should_escalate = self._should_escalate(masked_message, response_data.get('response', ''), sentiment, history)
            
            # Log to audit trail
            await self.secure_api.log_audit_trail(
                user_id=chat_message.user_id or "anonymous",
                action="chat_message",
                details={
                    "session_id": session_id,
                    "message_length": len(masked_message),
                    "detected_language": detected_language,
                    "intent": intent,
                    "sentiment": sentiment,
                    "similar_docs_found": len(similar_docs),
                    "should_escalate": should_escalate
                }
            )
            
            # Update conversation history
            history.append({
                "timestamp": datetime.now().isoformat(),
                "user_message": masked_message,
                "bot_response": response_data.get('response', ''),
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
                response=response_data.get('response', 'Maaf, terjadi kesalahan dalam memproses pesan Anda.'),
                session_id=session_id,
                detected_language=detected_language,
                intent=intent,
                sentiment=sentiment,
                confidence=0.9,  # Placeholder confidence score
                suggestions=suggestions,
                should_escalate=should_escalate,
                metadata={
                    "processing_time": processing_time,
                    "model_used": self.llm_client.model_id,
                    "tokens_used": response_data.get('tokens_used', 0),
                    "similar_docs_count": len(similar_docs),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return ChatResponse(
                response="Maaf, terjadi kesalahan sistem. Silakan coba lagi atau hubungi admin SIPD.",
                session_id=session_id,
                should_escalate=True,
                metadata={"error": str(e)}
            )
    
    async def _classify_intent(self, message: str) -> str:
        """Classify intent of message"""
        try:
            # Prepare prompt
            prompt = self.intent_classification_prompt.format(message=message)
            
            # Generate response
            response = await self.llm_client.generate_response(
                prompt=prompt,
                max_tokens=50,
                temperature=0.3
            )
            
            # Extract intent
            intent_text = response.get('response', '').strip().lower()
            
            # Map to standard intents
            intent_mapping = {
                "login": "login_issue",
                "login_issue": "login_issue",
                "masalah login": "login_issue",
                "dpa": "dpa_issue",
                "dpa_issue": "dpa_issue",
                "masalah dpa": "dpa_issue",
                "anggaran": "dpa_issue",
                "laporan": "laporan_issue",
                "laporan_issue": "laporan_issue",
                "masalah laporan": "laporan_issue",
                "report": "laporan_issue",
                "pertanyaan": "general_question",
                "general_question": "general_question",
                "pertanyaan umum": "general_question",
                "other": "other",
                "lainnya": "other"
            }
            
            # Find matching intent
            for key, value in intent_mapping.items():
                if key in intent_text:
                    return value
            
            # Default to other
            return "other"
            
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return "other"
    
    async def _analyze_sentiment(self, message: str) -> str:
        """Analyze sentiment of message"""
        try:
            # Prepare prompt
            prompt = self.sentiment_analysis_prompt.format(message=message)
            
            # Generate response
            response = await self.llm_client.generate_response(
                prompt=prompt,
                max_tokens=50,
                temperature=0.3
            )
            
            # Extract sentiment
            sentiment_text = response.get('response', '').strip().lower()
            
            # Map to standard sentiments
            if "positive" in sentiment_text or "positif" in sentiment_text:
                return "positive"
            elif "negative" in sentiment_text or "negatif" in sentiment_text:
                return "negative"
            else:
                return "neutral"
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return "neutral"
    
    async def _generate_response(
        self, 
        message: str, 
        language: str, 
        intent: str, 
        sentiment: str, 
        context: str, 
        history: List[Dict]
    ) -> Dict[str, Any]:
        """Generate response based on message, language, intent, sentiment, and context"""
        try:
            # Select appropriate system prompt based on language
            system_prompt = self.system_prompts.get(language, self.system_prompts['default'])
            
            # Build conversation context
            conversation_context = ""
            if history:
                # Get last 3 conversations
                recent_history = history[-3:] if len(history) > 3 else history
                
                if language == "id":
                    conversation_context = "\n\nRiwayat percakapan terakhir:\n"
                elif language == "en":
                    conversation_context = "\n\nRecent conversation history:\n"
                elif language == "jv":
                    conversation_context = "\n\nRiwayat pacelathon pungkasan:\n"
                elif language == "su":
                    conversation_context = "\n\nRiwayat obrolan panungtungan:\n"
                elif language == "ms":
                    conversation_context = "\n\nSejarah perbualan terkini:\n"
                else:
                    conversation_context = "\n\nRiwayat percakapan terakhir:\n"
                
                for entry in recent_history:
                    conversation_context += f"User: {entry.get('user_message', '')}\n"
                    conversation_context += f"Bot: {entry.get('bot_response', '')}\n"
            
            # Build full prompt
            full_prompt = f"{system_prompt}\n\n"
            
            # Add user context
            if language == "id":
                full_prompt += f"Intent terdeteksi: {intent}\nSentiment: {sentiment}\n\n"
            elif language == "en":
                full_prompt += f"Detected intent: {intent}\nSentiment: {sentiment}\n\n"
            else:
                full_prompt += f"Intent: {intent}\nSentiment: {sentiment}\n\n"
            
            # Add knowledge base context if available
            if context:
                if language == "id":
                    full_prompt += f"Informasi relevan dari knowledge base:\n{context}\n\n"
                elif language == "en":
                    full_prompt += f"Relevant information from knowledge base:\n{context}\n\n"
                elif language == "jv":
                    full_prompt += f"Informasi relevan saking knowledge base:\n{context}\n\n"
                elif language == "su":
                    full_prompt += f"Informasi relevan tina knowledge base:\n{context}\n\n"
                elif language == "ms":
                    full_prompt += f"Maklumat relevan daripada pangkalan pengetahuan:\n{context}\n\n"
                else:
                    full_prompt += f"Informasi relevan dari knowledge base:\n{context}\n\n"
            
            # Add conversation history
            full_prompt += f"{conversation_context}\n\n"
            
            # Add current message
            full_prompt += f"User: {message}\n\nBot:"
            
            # Generate response
            response_data = await self.llm_client.generate_response(
                prompt=full_prompt,
                system_prompt=system_prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            # Fallback responses based on language
            fallback_responses = {
                "id": "Maaf, saya mengalami kesulitan memproses permintaan Anda. Silakan coba lagi.",
                "en": "Sorry, I'm having trouble processing your request. Please try again.",
                "jv": "Nyuwun pangapunten, kula ngalami kesulitan ngolah panyuwunan panjenengan. Mangga cobi malih.",
                "su": "Punten, abdi ngalaman kasulitan ngolah pamundut anjeun. Mangga cobian deui.",
                "ms": "Maaf, saya mengalami kesukaran memproses permintaan anda. Sila cuba lagi."
            }
            
            return {
                "response": fallback_responses.get(language, fallback_responses["id"]),
                "error": str(e)
            }
    
    def _generate_suggestions(self, intent: str, language: str) -> List[str]:
        """Generate contextual suggestions based on intent and language"""
        # Suggestions for Indonesian
        suggestions_id = {
            "login_issue": [
                "Reset password",
                "Cek koneksi internet",
                "Hubungi admin"
            ],
            "dpa_issue": [
                "Cek format data",
                "Validasi field",
                "Refresh halaman"
            ],
            "laporan_issue": [
                "Cek periode",
                "Kurangi data",
                "Hubungi teknis"
            ],
            "general_question": [
                "Masalah login",
                "Masalah DPA",
                "Masalah laporan"
            ],
            "other": [
                "Masalah login",
                "Masalah DPA",
                "Masalah laporan"
            ]
        }
        
        # Suggestions for English
        suggestions_en = {
            "login_issue": [
                "Reset password",
                "Check internet connection",
                "Contact admin"
            ],
            "dpa_issue": [
                "Check data format",
                "Validate fields",
                "Refresh page"
            ],
            "laporan_issue": [
                "Check period",
                "Reduce data",
                "Contact technical support"
            ],
            "general_question": [
                "Login issues",
                "DPA issues",
                "Report issues"
            ],
            "other": [
                "Login issues",
                "DPA issues",
                "Report issues"
            ]
        }
        
        # Select suggestions based on language
        if language == "en":
            return suggestions_en.get(intent, suggestions_en["other"])
        else:
            return suggestions_id.get(intent, suggestions_id["other"])
    
    def _should_escalate(self, message: str, response: str, sentiment: str, history: List[Dict]) -> bool:
        """Determine if conversation should be escalated to human agent"""
        message_lower = message.lower()
        
        # Escalate if user explicitly asks for human
        if any(phrase in message_lower for phrase in ["human", "agent", "manusia", "operator", "admin"]):
            return True
        
        # Escalate if sentiment is negative
        if sentiment == "negative":
            # Check if this is the second consecutive negative sentiment
            if len(history) >= 1 and history[-1].get('sentiment') == "negative":
                return True
        
        # Escalate if conversation is going in circles
        if len(history) >= 3:
            # Check if user is repeating the same question
            recent_messages = [entry.get('user_message', '').lower() for entry in history[-3:]]
            if len(set(recent_messages)) == 1 and recent_messages[0] == message_lower:
                return True
        
        # Escalate if message contains urgent keywords
        urgent_keywords = ["urgent", "emergency", "darurat", "segera", "penting", "critical", "kritis"]
        if any(word in message_lower for word in urgent_keywords):
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
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of chatbot components"""
        try:
            # Get knowledge base stats
            kb_stats = await self.knowledge_embeddings.get_stats()
            
            # Get compliance status
            compliance_status = self.secure_api.get_compliance_status()
            
            # Get supported languages
            supported_languages = self.language_detector.get_supported_languages()
            
            return {
                "status": "healthy",
                "version": "2.0.0",
                "timestamp": datetime.now().isoformat(),
                "active_sessions": len(self.conversation_history),
                "knowledge_base": kb_stats,
                "compliance": compliance_status,
                "supported_languages": supported_languages,
                "llm_model": self.llm_client.model_id
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "status": "degraded",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Global chatbot instance
chatbot = EnhancedSIPDChatbot()

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Enhanced SIPD Chatbot...")
    await chatbot.initialize()
    logger.info("Enhanced SIPD Chatbot ready!")
    yield
    # Shutdown
    logger.info("Shutting down Enhanced SIPD Chatbot...")

# FastAPI app
app = FastAPI(
    title="Enhanced SIPD Chatbot",
    description="Chatbot SIPD dengan arsitektur canggih dan dukungan multilingual",
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
@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Halaman chat interface"""
    return """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Enhanced SIPD Chatbot</title>
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
                overflow: hidden;
                display: flex;
                flex-direction: column;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            
            .chat-header {
                background: #2c3e50;
                color: white;
                padding: 20px;
                text-align: center;
                position: relative;
            }
            
            .language-selector {
                position: absolute;
                right: 20px;
                top: 20px;
            }
            
            .language-selector select {
                padding: 5px 10px;
                border-radius: 5px;
                border: none;
                background: #34495e;
                color: white;
                cursor: pointer;
            }
            
            .chat-messages {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                background: #f8f9fa;
            }
            
            .message {
                margin: 10px 0;
                padding: 12px 16px;
                border-radius: 18px;
                max-width: 70%;
                word-wrap: break-word;
                position: relative;
            }
            
            .user-message {
                background: #007bff;
                color: white;
                margin-left: auto;
                text-align: right;
            }
            
            .bot-message {
                background: #e9ecef;
                color: #333;
                margin-right: auto;
                white-space: pre-line;
            }
            
            .suggestions {
                display: flex;
                flex-wrap: wrap;
                margin-top: 10px;
                justify-content: flex-start;
            }
            
            .suggestion-chip {
                background: #e3f2fd;
                color: #1976d2;
                padding: 8px 16px;
                margin: 4px;
                border-radius: 20px;
                font-size: 14px;
                cursor: pointer;
                border: 1px solid #bbdefb;
                transition: all 0.2s ease;
            }
            
            .suggestion-chip:hover {
                background: #bbdefb;
                transform: scale(1.05);
            }
            
            .chat-input {
                padding: 20px;
                background: white;
                border-top: 1px solid #dee2e6;
                display: flex;
                align-items: center;
            }
            
            .chat-input input {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #dee2e6;
                border-radius: 25px;
                outline: none;
                font-size: 16px;
                transition: border-color 0.2s;
            }
            
            .chat-input input:focus {
                border-color: #007bff;
            }
            
            .chat-input button {
                padding: 12px 24px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 25px;
                margin-left: 10px;
                cursor: pointer;
                font-weight: bold;
                transition: background 0.2s;
            }
            
            .chat-input button:hover {
                background: #0056b3;
            }
            
            .typing-indicator {
                font-style: italic;
                color: #666;
                margin-left: 10px;
            }
            
            .escalation-notice {
                background: #ffc107;
                color: #333;
                padding: 10px;
                text-align: center;
                font-weight: bold;
                margin-top: 10px;
                border-radius: 5px;
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
                <h1>🤖 Enhanced SIPD Chatbot</h1>
                <p>Asisten Virtual Cerdas untuk Help Desk SIPD</p>
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
                    Halo! Saya adalah asisten virtual SIPD versi terbaru. Saya siap membantu Anda menyelesaikan masalah teknis dan menjawab pertanyaan seputar SIPD.
                    
                    Coba tanyakan tentang:
                    • Masalah login
                    • Masalah DPA/anggaran
                    • Masalah laporan
                </div>
            </div>
            <div class="chat-input">
                <input type="text" id="messageInput" placeholder="Ketik pesan Anda di sini..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()">Kirim</button>
                <span id="typingIndicator" class="typing-indicator" style="display: none;">Mengetik...</span>
            </div>
        </div>

        <script>
            let sessionId = generateSessionId();
            let currentLanguage = 'id';
            
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
                    'id': 'Ketik pesan Anda di sini...',
                    'en': 'Type your message here...',
                    'jv': 'Ketik pesen panjenengan ing kene...',
                    'su': 'Ketik pesen anjeun di dieu...',
                    'ms': 'Taip mesej anda di sini...'
                };
                
                const buttonTexts = {
                    'id': 'Kirim',
                    'en': 'Send',
                    'jv': 'Kirim',
                    'su': 'Kirim',
                    'ms': 'Hantar'
                };
                
                messageInput.placeholder = placeholders[currentLanguage] || placeholders['id'];
                sendButton.textContent = buttonTexts[currentLanguage] || buttonTexts['id'];
                
                // Add welcome message in selected language
                const welcomeMessages = {
                    'id': 'Bahasa telah diubah ke Bahasa Indonesia.',
                    'en': 'Language has been changed to English.',
                    'jv': 'Basa sampun diganti dados Basa Jawa.',
                    'su': 'Basa geus dirobah jadi Basa Sunda.',
                    'ms': 'Bahasa telah ditukar kepada Bahasa Melayu.'
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
                        metadataText += `Waktu proses: ${metadata.processing_time.toFixed(2)}s | `;
                    }
                    if (metadata.model_used) {
                        metadataText += `Model: ${metadata.model_used}`;
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
                    'id': 'Masalah Anda akan dieskalasi ke agen manusia. Mohon tunggu sebentar.',
                    'en': 'Your issue will be escalated to a human agent. Please wait a moment.',
                    'jv': 'Masalah panjenengan badhe dipun-eskalasi dhateng agen manungsa. Mangga nengga sekedhap.',
                    'su': 'Masalah anjeun bakal dieskalasi ka agen manusa. Mangga antosan sakedap.',
                    'ms': 'Masalah anda akan diangkat kepada ejen manusia. Sila tunggu sebentar.'
                };
                
                noticeDiv.textContent = escalationMessages[language] || escalationMessages['id'];
                
                chatMessages.appendChild(noticeDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        </script>
    </body>
    </html>
    """

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage, background_tasks: BackgroundTasks):
    """Main chat endpoint"""
    response = await chatbot.process_message(message)
    
    # Log chat in background
    background_tasks.add_task(
        chatbot.secure_api.log_audit_trail,
        user_id=message.user_id or "anonymous",
        action="chat_api_call",
        details={
            "session_id": response.session_id,
            "message_length": len(message.message),
            "response_length": len(response.response),
            "detected_language": response.detected_language,
            "intent": response.intent,
            "sentiment": response.sentiment,
            "should_escalate": response.should_escalate
        }
    )
    
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return await chatbot.get_health_status()

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
    return {"languages": chatbot.language_detector.get_supported_languages()}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced SIPD Chatbot...")
    print("🌐 Access the chatbot at: http://localhost:8000")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), "data"), exist_ok=True)
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)
    
    uvicorn.run(
        "enhanced_chatbot_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )