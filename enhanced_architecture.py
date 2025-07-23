# Enhanced Architecture for SIPD AI Chatbot
# Implementasi arsitektur multilingual dengan Meta-LLaMA-3.1-70B-Instruct

import os
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import httpx
from loguru import logger
from datetime import datetime
import uuid
from contextlib import asynccontextmanager

# Import konfigurasi
from config import settings

# Setup logging
logger.add("logs/enhanced_chatbot.log", rotation="10 MB", level="INFO")

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

class LanguageDetector:
    """Komponen untuk deteksi bahasa"""
    
    def __init__(self):
        self.llm_client = None
        self.supported_languages = [
            "id", # Indonesian
            "en", # English
            "jv", # Javanese
            "su", # Sundanese
            "ms"  # Malay
        ]
    
    async def initialize(self):
        """Initialize language detector"""
        # Placeholder for actual initialization
        logger.info("Language detector initialized")
        return True
    
    async def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        # Placeholder implementation - in real system would use LLM or dedicated language detection model
        # For demo, we'll use a simple keyword-based approach
        
        text_lower = text.lower()
        
        # Indonesian keywords
        if any(word in text_lower for word in ["apa", "bagaimana", "tolong", "mohon", "tidak", "bisa", "silakan"]):
            return "id"
        # English keywords
        elif any(word in text_lower for word in ["what", "how", "please", "help", "can", "could", "thank"]):
            return "en"
        # Default to Indonesian
        return "id"

class MetaLLMClient:
    """Client for Meta-LLaMA-3.1-70B-Instruct model"""
    
    def __init__(self):
        self.api_key = os.getenv("META_LLM_API_KEY", "")
        self.base_url = os.getenv("META_LLM_BASE_URL", "https://api.llama.ai/v1")
        self.model_id = "meta-llama-3.1-70b-instruct"
        self.max_retries = 3
        self.timeout = 60
    
    async def initialize(self):
        """Initialize LLM client"""
        # Placeholder for actual initialization
        logger.info(f"Meta LLM client initialized with model: {self.model_id}")
        return True
    
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop_sequences: List[str] = None
    ) -> Dict[str, Any]:
        """Generate response from Meta LLM"""
        try:
            # Placeholder implementation - in real system would call Meta LLM API
            # For demo, we'll simulate a response
            
            # Simulate API call delay
            await asyncio.sleep(1)
            
            # Simple response generation based on prompt keywords
            prompt_lower = prompt.lower()
            
            if "login" in prompt_lower or "masuk" in prompt_lower:
                response = "Untuk masalah login SIPD, silakan coba langkah berikut:\n1. Pastikan username dan password benar\n2. Clear cache browser\n3. Coba browser lain\n4. Hubungi admin jika masih bermasalah"
            elif "dpa" in prompt_lower or "anggaran" in prompt_lower:
                response = "Untuk masalah DPA, pastikan:\n1. Semua field mandatory terisi\n2. Format data sesuai\n3. Koneksi internet stabil\n4. Refresh halaman dan coba lagi"
            elif "laporan" in prompt_lower or "report" in prompt_lower:
                response = "Untuk masalah laporan:\n1. Cek format data\n2. Pastikan periode laporan benar\n3. Coba export dengan data lebih sedikit\n4. Hubungi tim teknis jika error berlanjut"
            else:
                response = "Terima kasih atas pertanyaan Anda. Saya adalah asisten virtual SIPD yang siap membantu menyelesaikan masalah teknis. Bisa Anda jelaskan lebih detail masalah yang Anda hadapi?"
            
            return {
                "response": response,
                "tokens_used": len(prompt.split()) + len(response.split()),
                "model": self.model_id,
                "finish_reason": "stop"
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "Maaf, saya mengalami kesulitan memproses permintaan Anda. Silakan coba lagi.",
                "error": str(e)
            }

class PersonalizedKnowledgeEmbeddings:
    """Personalized Knowledge Embeddings untuk domain-specific data"""
    
    def __init__(self, collection_name: str = "sipd_knowledge_base"):
        self.collection_name = collection_name
        self.vector_store = None
        self.embedding_model = None
    
    async def initialize(self):
        """Initialize embedding system"""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            from sentence_transformers import SentenceTransformer
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize ChromaDB client
            chroma_client = chromadb.PersistentClient(
                path=settings.vector_db_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection
            self.vector_store = chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "SIPD Knowledge Base with Personalized Embeddings"}
            )
            
            logger.info(f"Personalized Knowledge Embeddings initialized with collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            return False
    
    async def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if not self.vector_store or not self.embedding_model:
                await self.initialize()
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Query the vector store
            results = self.vector_store.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            similar_docs = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score
                    similarity = 1 - distance
                    
                    similar_docs.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity': similarity,
                        'rank': i + 1
                    })
            
            logger.info(f"Found {len(similar_docs)} similar documents for query: {query[:50]}...")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []

class SecureAPILayer:
    """Secure API Layer with data masking and GRC features"""
    
    def __init__(self):
        self.audit_trail = []
    
    def mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive data like passwords, NIK, etc."""
        # Placeholder implementation - in real system would use regex patterns
        masked_text = text
        
        # Mask potential passwords
        import re
        password_pattern = r'\b(?:password|kata sandi|katasandi|pwd)\s*[:=]\s*([^\s,;]+)'  
        masked_text = re.sub(password_pattern, r'\1 [MASKED]', masked_text, flags=re.IGNORECASE)
        
        # Mask potential NIK (16 digit number)
        nik_pattern = r'\b\d{16}\b'
        masked_text = re.sub(nik_pattern, '[NIK MASKED]', masked_text)
        
        return masked_text
    
    def log_audit_trail(self, user_id: str, action: str, details: Dict[str, Any]):
        """Log audit trail for compliance"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "user_id": user_id,
            "action": action,
            "details": details
        }
        self.audit_trail.append(log_entry)
        logger.info(f"Audit trail: {action} by {user_id} at {timestamp}")
    
    def check_access_permission(self, user_id: str, resource: str) -> bool:
        """Check if user has permission to access resource"""
        # Placeholder implementation - in real system would check against a permission database
        return True

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
            "default": """Anda adalah asisten AI untuk Sistem Informasi Pemerintah Daerah (SIPD).
Berikan jawaban yang akurat, ramah, dan profesional."""
        }
    
    async def initialize(self):
        """Initialize all components"""
        try:
            await self.language_detector.initialize()
            await self.llm_client.initialize()
            await self.knowledge_embeddings.initialize()
            
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
            
            # Get relevant context from knowledge base
            similar_docs = await self.knowledge_embeddings.search_similar(masked_message)
            context = "\n\n---\n\n".join([doc['content'] for doc in similar_docs[:3]]) if similar_docs else ""
            
            # Select appropriate system prompt based on language
            system_prompt = self.system_prompts.get(detected_language, self.system_prompts['default'])
            
            # Build conversation context
            conversation_context = ""
            if history:
                conversation_context = "\n\nRiwayat percakapan terakhir:\n"
                for entry in history[-3:]:  # Last 3 conversations
                    conversation_context += f"User: {entry.get('user_message', '')}\n"
                    conversation_context += f"Bot: {entry.get('bot_response', '')}\n"
            
            # Build full prompt
            full_prompt = f"{system_prompt}\n\n"
            if context:
                full_prompt += f"Informasi relevan dari knowledge base:\n{context}\n\n"
            full_prompt += f"{conversation_context}\n\nUser: {masked_message}\n\nBot:"
            
            # Generate response
            response_data = await self.llm_client.generate_response(
                prompt=full_prompt,
                system_prompt=system_prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Generate suggestions based on context and response
            suggestions = self._generate_suggestions(masked_message, response_data.get('response', ''), detected_language)
            
            # Determine if escalation is needed
            should_escalate = self._should_escalate(masked_message, response_data.get('response', ''), history)
            
            # Log to audit trail
            self.secure_api.log_audit_trail(
                user_id=chat_message.user_id or "anonymous",
                action="chat_message",
                details={
                    "session_id": session_id,
                    "message_length": len(masked_message),
                    "detected_language": detected_language,
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
                intent=None,  # Would be determined by a dedicated intent classifier in a real system
                sentiment=None,  # Would be determined by a sentiment analyzer in a real system
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
    
    def _generate_suggestions(self, message: str, response: str, language: str) -> List[str]:
        """Generate contextual suggestions based on message and response"""
        message_lower = message.lower()
        
        # Indonesian suggestions
        if language == "id":
            if any(word in message_lower for word in ["login", "masuk", "akses"]):
                return [
                    "Reset password",
                    "Cek koneksi internet",
                    "Hubungi admin"
                ]
            elif any(word in message_lower for word in ["dpa", "anggaran", "input"]):
                return [
                    "Cek format data",
                    "Validasi field",
                    "Refresh halaman"
                ]
            elif any(word in message_lower for word in ["laporan", "export", "excel"]):
                return [
                    "Cek periode",
                    "Kurangi data",
                    "Hubungi teknis"
                ]
            else:
                return [
                    "Masalah login",
                    "Masalah DPA",
                    "Masalah laporan"
                ]
        # English suggestions
        elif language == "en":
            if any(word in message_lower for word in ["login", "access", "password"]):
                return [
                    "Reset password",
                    "Check internet connection",
                    "Contact admin"
                ]
            elif any(word in message_lower for word in ["dpa", "budget", "input"]):
                return [
                    "Check data format",
                    "Validate fields",
                    "Refresh page"
                ]
            elif any(word in message_lower for word in ["report", "export", "excel"]):
                return [
                    "Check period",
                    "Reduce data",
                    "Contact technical support"
                ]
            else:
                return [
                    "Login issues",
                    "DPA issues",
                    "Report issues"
                ]
        # Default suggestions
        else:
            return [
                "Login issues",
                "DPA issues",
                "Report issues"
            ]
    
    def _should_escalate(self, message: str, response: str, history: List[Dict]) -> bool:
        """Determine if conversation should be escalated to human agent"""
        message_lower = message.lower()
        
        # Escalate if user explicitly asks for human
        if any(phrase in message_lower for phrase in ["human", "agent", "manusia", "operator", "admin"]):
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
                if (currentLanguage === 'id') {
                    messageInput.placeholder = 'Ketik pesan Anda di sini...';
                } else {
                    messageInput.placeholder = 'Type your message here...';
                }
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
                    addMessage(data.response, 'bot');
                    
                    // Add escalation notice if needed
                    if (data.should_escalate) {
                        addEscalationNotice();
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
            
            function addMessage(message, sender) {
                const chatMessages = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                messageDiv.textContent = message;
                
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
            
            function addEscalationNotice() {
                const chatMessages = document.getElementById('chatMessages');
                const noticeDiv = document.createElement('div');
                noticeDiv.className = 'escalation-notice';
                
                if (currentLanguage === 'id') {
                    noticeDiv.textContent = 'Masalah Anda akan dieskalasi ke agen manusia. Mohon tunggu sebentar.';
                } else {
                    noticeDiv.textContent = 'Your issue will be escalated to a human agent. Please wait a moment.';
                }
                
                chatMessages.appendChild(noticeDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        </script>
    </body>
    </html>
    """

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Main chat endpoint"""
    return await chatbot.process_message(message)

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
    history = chatbot.conversation_history.get(session_id, [])
    return {"session_id": session_id, "history": history}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced SIPD Chatbot...")
    print("🌐 Access the chatbot at: http://localhost:8000")
    
    uvicorn.run(
        "enhanced_architecture:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )