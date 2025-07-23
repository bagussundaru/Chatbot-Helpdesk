import os
from typing import Optional, Dict, Any, List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

class Settings(BaseSettings):
    """Configuration settings for Enhanced SIPD AI Chatbot"""
    
    # Nebius AI Studio Configuration (legacy support)
    nebius_api_key: str = os.getenv("NEBIUS_API_KEY", "")
    nebius_base_url: str = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1")
    nebius_model_id: str = os.getenv("NEBIUS_MODEL_ID", "")
    
    # Meta LLM Configuration (new)
    meta_llm_api_key: Optional[str] = os.getenv("META_LLM_API_KEY", "")
    meta_llm_api_url: str = os.getenv("META_LLM_API_URL", "https://api.meta.ai/llama/v1/completions")
    meta_llm_model_id: str = os.getenv("META_LLM_MODEL_ID", "meta-llama-3.1-70b-instruct")
    meta_llm_timeout: int = int(os.getenv("META_LLM_TIMEOUT", "30"))
    meta_llm_max_retries: int = int(os.getenv("META_LLM_MAX_RETRIES", "3"))
    
    # Database Configuration
    database_url: str = os.getenv("DATABASE_URL", "postgresql://localhost:5432/sipd_chatbot")
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    
    # Application Configuration
    app_name: str = os.getenv("APP_NAME", "Enhanced SIPD AI Chatbot")
    app_version: str = os.getenv("APP_VERSION", "2.0.0")
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Modal.com Configuration (optional)
    modal_token_id: str = os.getenv("MODAL_TOKEN_ID", "")
    modal_token_secret: str = os.getenv("MODAL_TOKEN_SECRET", "")
    
    # Chatbot Configuration
    max_conversation_history: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "50"))
    response_timeout: int = int(os.getenv("RESPONSE_TIMEOUT", "30"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1000"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    max_message_length: int = int(os.getenv("MAX_MESSAGE_LENGTH", "4000"))
    
    # RAG Configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    collection_name: str = os.getenv("COLLECTION_NAME", "sipd_knowledge")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    embedding_cache_size: int = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))
    
    # Security settings
    secret_key: str = os.getenv("SECRET_KEY", "sipd-ai-chatbot-secret-key-change-in-production")
    audit_log_path: str = os.getenv("AUDIT_LOG_PATH", "./logs/audit_trail.jsonl")
    enable_data_masking: bool = os.getenv("ENABLE_DATA_MASKING", "True").lower() == "true"
    enable_audit_logging: bool = os.getenv("ENABLE_AUDIT_LOGGING", "True").lower() == "true"
    
    # Language detection settings
    default_language: str = os.getenv("DEFAULT_LANGUAGE", "id")
    supported_languages: List[str] = ["id", "en", "jv", "su", "ms"]
    language_detection_confidence_threshold: float = float(os.getenv("LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD", "0.7"))
    
    # Knowledge base settings
    knowledge_base_csv_path: str = os.getenv("KNOWLEDGE_BASE_CSV_PATH", "./data/sipd_knowledge_base.csv")
    
    # System prompts for different languages
    system_prompts: Dict[str, str] = {
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
    intent_classification_prompt: str = """Klasifikasikan intent dari pesan berikut ke dalam salah satu kategori: login_issue, dpa_issue, laporan_issue, general_question, atau other.

Pesan: {message}

Intent:"""
    
    # Sentiment analysis prompts
    sentiment_analysis_prompt: str = """Analisis sentiment dari pesan berikut dan kategorikan sebagai positive, neutral, atau negative.

Pesan: {message}

Sentiment:"""
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

# Setup logging
logger.remove()
logger.add(
    "logs/config.log",
    level=settings.log_level,
    rotation="10 MB",
    compression="zip"
)
logger.add(lambda msg: print(msg), level=settings.log_level)

# Create necessary directories
os.makedirs(os.path.dirname(settings.vector_db_path), exist_ok=True)
os.makedirs(os.path.dirname(settings.audit_log_path), exist_ok=True)

# Log configuration
if settings.debug:
    logger.debug(f"Application settings: {settings.__dict__}")
else:
    # Log non-sensitive settings only
    safe_settings = settings.__dict__.copy()
    for key in ["nebius_api_key", "meta_llm_api_key", "secret_key", "modal_token_id", "modal_token_secret"]:
        if key in safe_settings:
            safe_settings[key] = "[REDACTED]"
    logger.info(f"Application settings loaded: {safe_settings}")

# Export settings
__all__ = ["settings"]