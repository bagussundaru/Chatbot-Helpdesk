from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import time
import json

# Initialize FastAPI app
app = FastAPI(
    title="SIPD AI Chatbot",
    version="1.0.0",
    description="SIPD AI Chatbot - Intelligent Help Desk Assistant (Demo Version)"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage
conversation_history = {}
sample_responses = {
    "login": "Untuk masalah login SIPD, silakan coba langkah berikut:\n1. Pastikan username dan password benar\n2. Clear cache browser\n3. Coba browser lain\n4. Hubungi admin jika masih bermasalah",
    "dpa": "Untuk masalah DPA, pastikan:\n1. Semua field mandatory terisi\n2. Format data sesuai\n3. Koneksi internet stabil\n4. Refresh halaman dan coba lagi",
    "laporan": "Untuk masalah laporan:\n1. Cek format data\n2. Pastikan periode laporan benar\n3. Coba export dengan data lebih sedikit\n4. Hubungi tim teknis jika error berlanjut",
    "default": "Terima kasih atas pertanyaan Anda. Saya adalah asisten virtual SIPD yang siap membantu menyelesaikan masalah teknis. Bisa Anda jelaskan lebih detail masalah yang Anda hadapi?"
}

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    processing_time: float
    suggestions: List[str]
    timestamp: float

# Simple response generator
def generate_simple_response(message: str) -> Dict[str, Any]:
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["login", "masuk", "akses"]):
        response = sample_responses["login"]
        suggestions = ["Reset password", "Cek koneksi internet", "Hubungi admin"]
    elif any(word in message_lower for word in ["dpa", "anggaran", "input"]):
        response = sample_responses["dpa"]
        suggestions = ["Cek format data", "Validasi field", "Refresh halaman"]
    elif any(word in message_lower for word in ["laporan", "export", "excel"]):
        response = sample_responses["laporan"]
        suggestions = ["Cek periode", "Kurangi data", "Hubungi teknis"]
    else:
        response = sample_responses["default"]
        suggestions = ["Masalah login", "Masalah DPA", "Masalah laporan"]
    
    return {
        "response": response,
        "suggestions": suggestions
    }

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with simple chat interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SIPD AI Chatbot - Demo</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            .header {
                background: #2c3e50;
                color: white;
                padding: 20px;
                text-align: center;
            }
            .demo-notice {
                background: #f39c12;
                color: white;
                padding: 10px;
                text-align: center;
                font-weight: bold;
            }
            .chat-container {
                height: 500px;
                overflow-y: auto;
                padding: 20px;
                background: #f8f9fa;
            }
            .message {
                margin: 10px 0;
                padding: 12px 16px;
                border-radius: 18px;
                max-width: 70%;
                word-wrap: break-word;
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
            .input-container {
                padding: 20px;
                background: white;
                border-top: 1px solid #dee2e6;
            }
            .input-group {
                display: flex;
                gap: 10px;
            }
            #messageInput {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #dee2e6;
                border-radius: 25px;
                outline: none;
                font-size: 14px;
            }
            #messageInput:focus {
                border-color: #007bff;
            }
            #sendButton {
                padding: 12px 24px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-weight: bold;
            }
            #sendButton:hover {
                background: #0056b3;
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
                border-radius: 15px;
                font-size: 12px;
                cursor: pointer;
                border: 1px solid #bbdefb;
            }
            .suggestion-chip:hover {
                background: #bbdefb;
            }
            .typing {
                font-style: italic;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 SIPD AI Chatbot</h1>
                <p>Asisten Virtual Cerdas untuk Help Desk SIPD</p>
            </div>
            <div class="demo-notice">
                ⚠️ DEMO VERSION - Menggunakan respons template sederhana
            </div>
            <div class="chat-container" id="chatContainer">
                <div class="message bot-message">
                    Halo! Saya adalah asisten virtual SIPD versi demo. Saya siap membantu Anda menyelesaikan masalah teknis dan menjawab pertanyaan seputar SIPD. 
                    
                    Coba tanyakan tentang:
                    • Masalah login
                    • Masalah DPA/anggaran
                    • Masalah laporan
                </div>
            </div>
            <div class="input-container">
                <div class="input-group">
                    <input type="text" id="messageInput" placeholder="Ketik pesan Anda di sini..." onkeypress="handleKeyPress(event)">
                    <button id="sendButton" onclick="sendMessage()">Kirim</button>
                </div>
            </div>
        </div>

        <script>
            let sessionId = generateSessionId();
            
            function generateSessionId() {
                return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
            
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                
                if (!message) return;
                
                // Add user message to chat
                addMessageToChat(message, 'user');
                input.value = '';
                
                // Show typing indicator
                const typingDiv = addMessageToChat('Sedang mengetik...', 'bot', true);
                
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
                    
                    // Remove typing indicator
                    typingDiv.remove();
                    
                    // Add bot response
                    addMessageToChat(data.response, 'bot');
                    
                    // Add suggestions if available
                    if (data.suggestions && data.suggestions.length > 0) {
                        addSuggestions(data.suggestions);
                    }
                    
                } catch (error) {
                    typingDiv.remove();
                    addMessageToChat('Maaf, terjadi kesalahan. Silakan coba lagi.', 'bot');
                    console.error('Error:', error);
                }
            }
            
            function addMessageToChat(message, sender, isTyping = false) {
                const chatContainer = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message${isTyping ? ' typing' : ''}`;
                messageDiv.textContent = message;
                
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
                
                return messageDiv;
            }
            
            function addSuggestions(suggestions) {
                const chatContainer = document.getElementById('chatContainer');
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
                
                chatContainer.appendChild(suggestionsDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Main chat endpoint"""
    start_time = time.time()
    
    # Generate session ID if not provided
    if not message.session_id:
        message.session_id = str(uuid.uuid4())
    
    # Generate simple response
    result = generate_simple_response(message.message)
    
    # Update conversation history
    if message.session_id not in conversation_history:
        conversation_history[message.session_id] = []
    
    conversation_history[message.session_id].append({
        "user": message.message,
        "assistant": result["response"],
        "timestamp": time.time()
    })
    
    # Prepare response
    response = ChatResponse(
        response=result["response"],
        session_id=message.session_id,
        processing_time=round(time.time() - start_time, 2),
        suggestions=result["suggestions"],
        timestamp=time.time()
    )
    
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "demo",
        "timestamp": time.time(),
        "active_sessions": len(conversation_history)
    }

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get conversation history for a session"""
    history = conversation_history.get(session_id, [])
    return {"session_id": session_id, "history": history}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting SIPD AI Chatbot Demo...")
    print("📝 This is a simplified demo version")
    print("🌐 Access the chatbot at: http://localhost:8000")
    
    uvicorn.run(
        "simple_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )