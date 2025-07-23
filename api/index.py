from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import re
import os
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    message: str
    language: str = "id"

class ChatResponse(BaseModel):
    response: str
    suggestions: List[str] = []
    escalate: bool = False
    language: str = "id"

class SimplifiedEnhancedChatbot:
    def __init__(self):
        self.conversation_history = []
        self.system_prompts = {
            "id": "Anda adalah asisten helpdesk SIPD yang membantu dengan masalah sistem informasi perencanaan daerah.",
            "en": "You are a SIPD helpdesk assistant helping with regional planning information system issues."
        }
        
        # Sample responses for common issues
        self.sample_responses = {
            "id": {
                "login": "Untuk masalah login SIPD, pastikan username dan password benar. Jika masih bermasalah, hubungi admin sistem.",
                "akses": "Masalah akses biasanya terkait dengan hak akses pengguna. Silakan hubungi administrator untuk verifikasi akun Anda.",
                "data": "Untuk masalah data, pastikan koneksi internet stabil dan coba refresh halaman. Jika masih bermasalah, laporkan ke tim teknis.",
                "sistem": "Jika sistem lambat atau error, coba clear cache browser atau gunakan browser lain. Laporkan jika masalah berlanjut."
            },
            "en": {
                "login": "For SIPD login issues, ensure username and password are correct. If problems persist, contact system admin.",
                "access": "Access issues are usually related to user permissions. Please contact administrator to verify your account.",
                "data": "For data issues, ensure stable internet connection and try refreshing the page. Report to technical team if problems persist.",
                "system": "If system is slow or showing errors, try clearing browser cache or use different browser. Report if issues continue."
            }
        }
    
    def detect_language(self, text: str) -> str:
        """Simple language detection"""
        indonesian_words = ['dan', 'atau', 'dengan', 'untuk', 'dari', 'ke', 'di', 'pada', 'yang', 'adalah', 'ini', 'itu', 'saya', 'anda', 'tidak', 'bisa', 'masalah', 'sistem', 'login', 'akses', 'data']
        english_words = ['and', 'or', 'with', 'for', 'from', 'to', 'in', 'on', 'the', 'is', 'this', 'that', 'i', 'you', 'cannot', 'can', 'problem', 'system', 'login', 'access', 'data']
        
        text_lower = text.lower()
        id_count = sum(1 for word in indonesian_words if word in text_lower)
        en_count = sum(1 for word in english_words if word in text_lower)
        
        return "id" if id_count >= en_count else "en"
    
    def classify_intent(self, message: str, language: str) -> str:
        """Classify user intent"""
        message_lower = message.lower()
        
        if language == "id":
            if any(word in message_lower for word in ['login', 'masuk', 'password', 'username']):
                return "login"
            elif any(word in message_lower for word in ['akses', 'hak', 'permission', 'izin']):
                return "akses"
            elif any(word in message_lower for word in ['data', 'informasi', 'laporan']):
                return "data"
            elif any(word in message_lower for word in ['sistem', 'error', 'lambat', 'crash']):
                return "sistem"
        else:
            if any(word in message_lower for word in ['login', 'password', 'username', 'sign']):
                return "login"
            elif any(word in message_lower for word in ['access', 'permission', 'rights']):
                return "access"
            elif any(word in message_lower for word in ['data', 'information', 'report']):
                return "data"
            elif any(word in message_lower for word in ['system', 'error', 'slow', 'crash']):
                return "system"
        
        return "general"
    
    def generate_response(self, message: str, language: str) -> ChatResponse:
        """Generate response based on message and language"""
        intent = self.classify_intent(message, language)
        
        # Get base response
        if intent in self.sample_responses[language]:
            response = self.sample_responses[language][intent]
        else:
            if language == "id":
                response = "Terima kasih atas pertanyaan Anda. Tim helpdesk SIPD akan membantu menyelesaikan masalah Anda. Mohon berikan detail lebih lanjut tentang masalah yang Anda hadapi."
            else:
                response = "Thank you for your question. SIPD helpdesk team will help resolve your issue. Please provide more details about the problem you're experiencing."
        
        # Generate suggestions
        suggestions = []
        if language == "id":
            suggestions = [
                "Masalah Login SIPD",
                "Akses Data Perencanaan",
                "Error Sistem",
                "Bantuan Teknis Lainnya"
            ]
        else:
            suggestions = [
                "SIPD Login Issues",
                "Planning Data Access",
                "System Errors",
                "Other Technical Support"
            ]
        
        # Determine if escalation is needed
        escalate = any(word in message.lower() for word in ['urgent', 'penting', 'segera', 'critical', 'kritis'])
        
        return ChatResponse(
            response=response,
            suggestions=suggestions,
            escalate=escalate,
            language=language
        )

# Initialize chatbot
chatbot = SimplifiedEnhancedChatbot()

# Create FastAPI app
app = FastAPI(title="SIPD Helpdesk Chatbot")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/logo/SIPD.svg")
async def get_logo():
    """Serve the SIPD logo"""
    logo_path = os.path.join(os.path.dirname(__file__), "..", "logo", "SIPD.svg")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/svg+xml")
    else:
        # Return a simple SVG if file not found
        svg_content = '''<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
            <rect width="100" height="100" fill="#0066cc"/>
            <text x="50" y="55" font-family="Arial" font-size="16" fill="white" text-anchor="middle">SIPD</text>
        </svg>'''
        return HTMLResponse(content=svg_content, media_type="image/svg+xml")

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the chat interface"""
    html_content = '''
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIPD Helpdesk Chatbot</title>
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
            background: linear-gradient(135deg, #0066cc, #004499);
            color: white;
            padding: 20px;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }
        
        .chat-header img {
            width: 40px;
            height: 40px;
            border-radius: 8px;
            background: white;
            padding: 5px;
        }
        
        .chat-header h1 {
            font-size: 24px;
            font-weight: 600;
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
            gap: 10px;
        }
        
        .user-message {
            justify-content: flex-end;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
        }
        
        .user-message .message-content {
            background: #0066cc;
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .bot-message .message-content {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 4px;
        }
        
        .suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        
        .suggestion-btn {
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 20px;
            padding: 8px 16px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        
        .suggestion-btn:hover {
            background: #0066cc;
            color: white;
            border-color: #0066cc;
        }
        
        .escalation-notice {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 12px;
            margin-top: 10px;
            color: #856404;
        }
        
        .chat-input {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .message-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s ease;
        }
        
        .message-input:focus {
            border-color: #0066cc;
        }
        
        .send-btn {
            background: #0066cc;
            color: white;
            border: none;
            border-radius: 50%;
            width: 45px;
            height: 45px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background-color 0.3s ease;
        }
        
        .send-btn:hover {
            background: #0052a3;
        }
        
        .send-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .welcome-message {
            text-align: center;
            padding: 40px 20px;
            color: #666;
        }
        
        .welcome-message img {
            width: 80px;
            height: 80px;
            margin-bottom: 20px;
            border-radius: 12px;
            background: white;
            padding: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .welcome-message h2 {
            color: #333;
            margin-bottom: 10px;
            font-size: 24px;
        }
        
        .welcome-message p {
            font-size: 16px;
            line-height: 1.5;
        }
        
        .typing-indicator {
            display: none;
            padding: 10px;
            font-style: italic;
            color: #666;
        }
        
        @media (max-width: 768px) {
            .chat-container {
                width: 95%;
                height: 95vh;
                border-radius: 10px;
            }
            
            .chat-header {
                padding: 15px;
            }
            
            .chat-header h1 {
                font-size: 20px;
            }
            
            .message-content {
                max-width: 85%;
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <img src="/logo/SIPD.svg" alt="SIPD Logo">
            <h1>SIPD Helpdesk</h1>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="welcome-message">
                <img src="/logo/SIPD.svg" alt="SIPD Logo">
                <h2>Halo! Terima kasih sudah menghubungi Asisten Helpdesk SIPD</h2>
                <p>Saya siap membantu Anda dengan masalah sistem informasi perencanaan daerah. Silakan sampaikan pertanyaan atau masalah yang Anda hadapi.</p>
            </div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            Asisten sedang mengetik...
        </div>
        
        <div class="chat-input">
            <div class="input-container">
                <input type="text" class="message-input" id="messageInput" placeholder="Ketik pesan Anda di sini..." maxlength="500">
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">
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
        const sendBtn = document.getElementById('sendBtn');
        const typingIndicator = document.getElementById('typingIndicator');
        
        const welcomeMessages = {
            id: "Selamat datang di Asisten SIPD",
            en: "Welcome to SIPD Assistant"
        };
        
        let currentLanguage = 'id';
        
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            // Disable input and show typing
            messageInput.disabled = true;
            sendBtn.disabled = true;
            messageInput.value = '';
            
            // Add user message
            addMessage(message, 'user');
            
            // Show typing indicator
            typingIndicator.style.display = 'block';
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        language: currentLanguage
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                
                const data = await response.json();
                
                // Hide typing indicator
                typingIndicator.style.display = 'none';
                
                // Add bot response
                addMessage(data.response, 'bot', data.suggestions, data.escalate);
                
                // Update current language
                currentLanguage = data.language;
                
            } catch (error) {
                console.error('Error:', error);
                typingIndicator.style.display = 'none';
                addMessage('Maaf, terjadi kesalahan. Silakan coba lagi.', 'bot');
            }
            
            // Re-enable input
            messageInput.disabled = false;
            sendBtn.disabled = false;
            messageInput.focus();
        }
        
        function addMessage(content, sender, suggestions = [], escalate = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.textContent = content;
            
            messageDiv.appendChild(messageContent);
            
            // Add suggestions for bot messages
            if (sender === 'bot' && suggestions.length > 0) {
                addSuggestions(messageDiv, suggestions);
            }
            
            // Add escalation notice if needed
            if (escalate) {
                addEscalationNotice(messageDiv);
            }
            
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function addSuggestions(messageDiv, suggestions) {
            const suggestionsDiv = document.createElement('div');
            suggestionsDiv.className = 'suggestions';
            
            suggestions.forEach(suggestion => {
                const btn = document.createElement('button');
                btn.className = 'suggestion-btn';
                btn.textContent = suggestion;
                btn.onclick = () => {
                    messageInput.value = suggestion;
                    sendMessage();
                };
                suggestionsDiv.appendChild(btn);
            });
            
            messageDiv.appendChild(suggestionsDiv);
        }
        
        function addEscalationNotice(messageDiv) {
            const noticeDiv = document.createElement('div');
            noticeDiv.className = 'escalation-notice';
            noticeDiv.innerHTML = '<strong>Catatan:</strong> Masalah Anda telah ditandai sebagai prioritas tinggi dan akan segera ditangani oleh tim teknis.';
            messageDiv.appendChild(noticeDiv);
        }
        
        // Focus on input when page loads
        window.addEventListener('load', () => {
            messageInput.focus();
        });
    </script>
</body>
</html>
    '''
    return HTMLResponse(content=html_content)

@app.post("/chat")
async def chat_endpoint(message: ChatMessage):
    """Handle chat messages"""
    try:
        # Detect language if not provided
        if not message.language:
            message.language = chatbot.detect_language(message.message)
        
        # Generate response
        response = chatbot.generate_response(message.message, message.language)
        
        # Add to conversation history
        chatbot.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_message": message.message,
            "bot_response": response.response,
            "language": message.language
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/history")
async def get_conversation_history():
    """Get conversation history"""
    return {"history": chatbot.conversation_history[-10:]}  # Return last 10 conversations

@app.delete("/history")
async def clear_conversation_history():
    """Clear conversation history"""
    chatbot.conversation_history.clear()
    return {"message": "Conversation history cleared"}

@app.get("/languages")
async def get_supported_languages():
    """Get supported languages"""
    return {
        "supported_languages": [
            {"code": "id", "name": "Bahasa Indonesia"},
            {"code": "en", "name": "English"}
        ]
    }

# For Vercel
handler = app