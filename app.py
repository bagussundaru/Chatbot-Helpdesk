from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import time
from loguru import logger
from chatbot_engine import SIPDChatbotEngine
from config import settings

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="SIPD AI Chatbot - Intelligent Help Desk Assistant"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot engine
chatbot_engine = SIPDChatbotEngine()

# Pydantic models for request/response
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: Dict[str, Any]
    sentiment: Dict[str, Any]
    processing_time: float
    has_context: bool
    suggestions: List[str]
    timestamp: float

class FeedbackRequest(BaseModel):
    session_id: str
    message_index: int
    rating: int  # 1-5 scale
    comment: Optional[str] = ""

class EscalationRequest(BaseModel):
    session_id: str
    reason: Optional[str] = ""

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with simple chat interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SIPD AI Chatbot</title>
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
            <div class="chat-container" id="chatContainer">
                <div class="message bot-message">
                    Halo! Saya adalah asisten virtual SIPD. Saya siap membantu Anda menyelesaikan masalah teknis dan menjawab pertanyaan seputar SIPD. Silakan ceritakan masalah yang Anda hadapi.
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
    try:
        # Generate session ID if not provided
        if not message.session_id:
            message.session_id = str(uuid.uuid4())
        
        # Process message through chatbot engine
        result = chatbot_engine.process_message(
            user_message=message.message,
            session_id=message.session_id,
            user_context=message.user_context
        )
        
        # Add timestamp
        result["timestamp"] = time.time()
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get conversation history for a session"""
    try:
        history = chatbot_engine.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear conversation history for a session"""
    try:
        success = chatbot_engine.clear_session_history(session_id)
        return {"success": success, "message": "History cleared" if success else "Session not found"}
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback"""
    try:
        success = chatbot_engine.add_feedback(
            session_id=feedback.session_id,
            message_index=feedback.message_index,
            rating=feedback.rating,
            comment=feedback.comment
        )
        return {"success": success, "message": "Feedback submitted successfully"}
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/escalate")
async def escalate_to_human(escalation: EscalationRequest):
    """Escalate conversation to human agent"""
    try:
        result = chatbot_engine.escalate_to_human(
            session_id=escalation.session_id,
            reason=escalation.reason
        )
        return result
    except Exception as e:
        logger.error(f"Error in escalation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        stats = chatbot_engine.get_system_stats()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "system_stats": stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/stats")
async def get_system_stats():
    """Get detailed system statistics"""
    try:
        stats = chatbot_engine.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )