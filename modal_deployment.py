import modal
from modal import Image, App, web_endpoint
import os
from typing import Dict, Any

# Define the Modal app
app = App("sipd-chatbot")

# Create custom image with dependencies
image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "pydantic==2.5.0",
        "python-multipart==0.0.6",
        "openai==1.3.0",
        "langchain==0.0.350",
        "langchain-openai==0.0.2",
        "langchain-community==0.0.10",
        "sentence-transformers==2.2.2",
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "chromadb==0.4.18",
        "requests==2.31.0",
        "python-dotenv==1.0.0",
        "loguru==0.7.2",
        "typing-extensions==4.8.0"
    )
    .copy_local_file("config.py", "/app/config.py")
    .copy_local_file("data_processor.py", "/app/data_processor.py")
    .copy_local_file("rag_system.py", "/app/rag_system.py")
    .copy_local_file("nebius_client.py", "/app/nebius_client.py")
    .copy_local_file("chatbot_engine.py", "/app/chatbot_engine.py")
    .copy_local_file("app.py", "/app/app.py")
    .workdir("/app")
)

# Define secrets for environment variables
secrets = [
    modal.Secret.from_name("sipd-chatbot-secrets")  # Create this in Modal dashboard
]

# Volume for persistent storage (vector database)
vector_volume = modal.Volume.from_name("sipd-vector-store", create_if_missing=True)

@app.function(
    image=image,
    secrets=secrets,
    volumes={"./vector_store": vector_volume},
    cpu=2.0,
    memory=4096,
    timeout=300,
    container_idle_timeout=300,
    allow_concurrent_inputs=10
)
@web_endpoint(method="GET")
def root():
    """Root endpoint with chat interface"""
    from app import app as fastapi_app
    import uvicorn
    from fastapi.responses import HTMLResponse
    
    # Get the root response from FastAPI app
    from app import root as fastapi_root
    return fastapi_root()

@app.function(
    image=image,
    secrets=secrets,
    volumes={"./vector_store": vector_volume},
    cpu=2.0,
    memory=4096,
    timeout=60,
    container_idle_timeout=300,
    allow_concurrent_inputs=20
)
@web_endpoint(method="POST")
def chat(request_data: Dict[str, Any]):
    """Chat endpoint"""
    import sys
    sys.path.append('/app')
    
    from chatbot_engine import SIPDChatbotEngine
    import time
    import uuid
    
    try:
        # Initialize chatbot engine (cached in container)
        if not hasattr(chat, '_engine'):
            chat._engine = SIPDChatbotEngine()
        
        # Extract message data
        message = request_data.get('message', '')
        session_id = request_data.get('session_id') or str(uuid.uuid4())
        user_context = request_data.get('user_context')
        
        # Process message
        result = chat._engine.process_message(
            user_message=message,
            session_id=session_id,
            user_context=user_context
        )
        
        # Add timestamp
        result["timestamp"] = time.time()
        
        return result
        
    except Exception as e:
        return {
            "response": "Maaf, terjadi kesalahan dalam memproses pesan Anda. Silakan coba lagi.",
            "error": str(e),
            "session_id": session_id,
            "processing_time": 0
        }

@app.function(
    image=image,
    secrets=secrets,
    volumes={"./vector_store": vector_volume},
    cpu=1.0,
    memory=2048,
    timeout=30
)
@web_endpoint(method="GET")
def health():
    """Health check endpoint"""
    import sys
    sys.path.append('/app')
    
    try:
        from chatbot_engine import SIPDChatbotEngine
        import time
        
        # Quick health check
        if not hasattr(health, '_engine'):
            health._engine = SIPDChatbotEngine()
            
        stats = health._engine.get_system_stats()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "system_stats": stats
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

@app.function(
    image=image,
    secrets=secrets,
    volumes={"./vector_store": vector_volume},
    cpu=1.0,
    memory=2048,
    timeout=30
)
@web_endpoint(method="GET")
def chat_history(session_id: str):
    """Get chat history endpoint"""
    import sys
    sys.path.append('/app')
    
    try:
        from chatbot_engine import SIPDChatbotEngine
        
        if not hasattr(chat_history, '_engine'):
            chat_history._engine = SIPDChatbotEngine()
            
        history = chat_history._engine.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
        
    except Exception as e:
        return {"error": str(e)}

@app.function(
    image=image,
    secrets=secrets,
    volumes={"./vector_store": vector_volume},
    cpu=1.0,
    memory=2048,
    timeout=30
)
@web_endpoint(method="POST")
def feedback(request_data: Dict[str, Any]):
    """Submit feedback endpoint"""
    import sys
    sys.path.append('/app')
    
    try:
        from chatbot_engine import SIPDChatbotEngine
        
        if not hasattr(feedback, '_engine'):
            feedback._engine = SIPDChatbotEngine()
            
        success = feedback._engine.add_feedback(
            session_id=request_data.get('session_id'),
            message_index=request_data.get('message_index'),
            rating=request_data.get('rating'),
            comment=request_data.get('comment', '')
        )
        
        return {"success": success, "message": "Feedback submitted successfully"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.function(
    image=image,
    secrets=secrets,
    volumes={"./vector_store": vector_volume},
    cpu=1.0,
    memory=2048,
    timeout=30
)
@web_endpoint(method="POST")
def escalate(request_data: Dict[str, Any]):
    """Escalate to human endpoint"""
    import sys
    sys.path.append('/app')
    
    try:
        from chatbot_engine import SIPDChatbotEngine
        
        if not hasattr(escalate, '_engine'):
            escalate._engine = SIPDChatbotEngine()
            
        result = escalate._engine.escalate_to_human(
            session_id=request_data.get('session_id'),
            reason=request_data.get('reason', '')
        )
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Maaf, terjadi kesalahan dalam membuat tiket dukungan."
        }

@app.function(
    image=image,
    secrets=secrets,
    volumes={"./vector_store": vector_volume},
    cpu=2.0,
    memory=4096,
    timeout=600
)
def initialize_knowledge_base():
    """Initialize knowledge base from CSV files (run once)"""
    import sys
    sys.path.append('/app')
    
    try:
        from data_processor import SIPDDataProcessor
        from rag_system import SIPDRAGSystem
        
        # Process CSV data
        processor = SIPDDataProcessor("./data/csv")
        training_data = processor.process_all_data()
        
        # Initialize RAG system
        rag = SIPDRAGSystem()
        rag.initialize_knowledge_base("./data/csv")
        
        return {
            "success": True,
            "training_examples": len(training_data),
            "message": "Knowledge base initialized successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to initialize knowledge base"
        }

# Local development function
@app.local_entrypoint()
def main():
    """Local entrypoint for testing"""
    print("SIPD Chatbot Modal deployment ready!")
    print("Available endpoints:")
    print("- GET /: Chat interface")
    print("- POST /chat: Chat API")
    print("- GET /health: Health check")
    print("- GET /chat_history/{session_id}: Get chat history")
    print("- POST /feedback: Submit feedback")
    print("- POST /escalate: Escalate to human")
    
    # Test initialization
    result = initialize_knowledge_base.remote()
    print(f"Knowledge base initialization: {result}")

if __name__ == "__main__":
    main()