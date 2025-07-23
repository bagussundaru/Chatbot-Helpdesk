# Enhanced Configuration for Nebius Embedding Integration
# Additional settings for optimized AI performance

import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class NebiusEmbeddingSettings(BaseSettings):
    """Enhanced settings for Nebius AI Studio integration"""
    
    # Nebius AI Studio Configuration
    NEBIUS_API_KEY: str = Field(..., env="NEBIUS_API_KEY")
    NEBIUS_BASE_URL: str = Field(default="https://api.studio.nebius.ai/v1", env="NEBIUS_BASE_URL")
    NEBIUS_MODEL_ID: str = Field(default="mistral-7b-instruct", env="NEBIUS_MODEL_ID")
    
    # Embedding Model Configuration
    NEBIUS_EMBEDDING_MODEL: str = Field(default="text-embedding-ada-002", env="NEBIUS_EMBEDDING_MODEL")
    EMBEDDING_DIMENSION: int = Field(default=1536, env="EMBEDDING_DIMENSION")
    MAX_EMBEDDING_TOKENS: int = Field(default=8191, env="MAX_EMBEDDING_TOKENS")
    
    # Performance Optimization
    EMBEDDING_BATCH_SIZE: int = Field(default=10, env="EMBEDDING_BATCH_SIZE")
    EMBEDDING_CACHE_SIZE: int = Field(default=1000, env="EMBEDDING_CACHE_SIZE")
    EMBEDDING_TIMEOUT: int = Field(default=30, env="EMBEDDING_TIMEOUT")
    MAX_RETRIES: int = Field(default=3, env="MAX_RETRIES")
    RETRY_DELAY: float = Field(default=1.0, env="RETRY_DELAY")
    
    # RAG System Configuration
    SIMILARITY_THRESHOLD: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    MAX_CONTEXT_LENGTH: int = Field(default=2000, env="MAX_CONTEXT_LENGTH")
    TOP_K_RESULTS: int = Field(default=5, env="TOP_K_RESULTS")
    CHUNK_SIZE: int = Field(default=500, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=50, env="CHUNK_OVERLAP")
    
    # Vector Store Configuration
    VECTOR_STORE_PATH: str = Field(default="./vector_store", env="VECTOR_STORE_PATH")
    COLLECTION_NAME: str = Field(default="sipd_knowledge_base", env="COLLECTION_NAME")
    PERSIST_DIRECTORY: str = Field(default="./chroma_db", env="PERSIST_DIRECTORY")
    
    # Cost Optimization
    ENABLE_EMBEDDING_CACHE: bool = Field(default=True, env="ENABLE_EMBEDDING_CACHE")
    CACHE_TTL_HOURS: int = Field(default=24, env="CACHE_TTL_HOURS")
    ENABLE_COMPRESSION: bool = Field(default=True, env="ENABLE_COMPRESSION")
    
    # Monitoring and Logging
    LOG_EMBEDDING_USAGE: bool = Field(default=True, env="LOG_EMBEDDING_USAGE")
    TRACK_PERFORMANCE_METRICS: bool = Field(default=True, env="TRACK_PERFORMANCE_METRICS")
    ENABLE_DEBUG_LOGGING: bool = Field(default=False, env="ENABLE_DEBUG_LOGGING")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

class ChatbotOptimizationSettings(BaseSettings):
    """Optimization settings for chatbot performance"""
    
    # Response Generation
    MAX_RESPONSE_LENGTH: int = Field(default=1000, env="MAX_RESPONSE_LENGTH")
    RESPONSE_TEMPERATURE: float = Field(default=0.7, env="RESPONSE_TEMPERATURE")
    TOP_P: float = Field(default=0.9, env="TOP_P")
    FREQUENCY_PENALTY: float = Field(default=0.1, env="FREQUENCY_PENALTY")
    PRESENCE_PENALTY: float = Field(default=0.1, env="PRESENCE_PENALTY")
    
    # Context Management
    MAX_CONVERSATION_HISTORY: int = Field(default=10, env="MAX_CONVERSATION_HISTORY")
    CONTEXT_WINDOW_SIZE: int = Field(default=4000, env="CONTEXT_WINDOW_SIZE")
    ENABLE_CONTEXT_COMPRESSION: bool = Field(default=True, env="ENABLE_CONTEXT_COMPRESSION")
    
    # Intent Classification
    INTENT_CONFIDENCE_THRESHOLD: float = Field(default=0.8, env="INTENT_CONFIDENCE_THRESHOLD")
    ENABLE_MULTI_INTENT: bool = Field(default=True, env="ENABLE_MULTI_INTENT")
    
    # Sentiment Analysis
    SENTIMENT_THRESHOLD: float = Field(default=0.6, env="SENTIMENT_THRESHOLD")
    ENABLE_EMOTION_DETECTION: bool = Field(default=True, env="ENABLE_EMOTION_DETECTION")
    
    # Escalation Rules
    AUTO_ESCALATE_NEGATIVE_SENTIMENT: bool = Field(default=True, env="AUTO_ESCALATE_NEGATIVE_SENTIMENT")
    ESCALATION_SENTIMENT_THRESHOLD: float = Field(default=-0.5, env="ESCALATION_SENTIMENT_THRESHOLD")
    MAX_FAILED_ATTEMPTS: int = Field(default=3, env="MAX_FAILED_ATTEMPTS")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Predefined prompts for different scenarios
SIPD_PROMPTS = {
    "system_prompt": """
Anda adalah asisten AI yang ahli dalam sistem SIPD (Sistem Informasi Pemerintahan Daerah). 
Anda membantu pengguna menyelesaikan masalah teknis dan memberikan panduan penggunaan sistem.

Karakteristik respons Anda:
- Profesional namun ramah
- Memberikan solusi yang praktis dan mudah dipahami
- Empati terhadap kesulitan pengguna
- Menggunakan bahasa Indonesia yang baik dan benar
- Memberikan langkah-langkah yang jelas dan terstruktur

Jika tidak yakin dengan jawaban, arahkan pengguna untuk menghubungi tim support.
""",
    
    "context_prompt": """
Berdasarkan konteks berikut dari knowledge base SIPD:
{context}

Jawab pertanyaan pengguna dengan mengacu pada informasi di atas. 
Jika informasi tidak cukup, berikan saran umum yang relevan.
""",
    
    "intent_classification_prompt": """
Klasifikasikan intent dari pesan pengguna berikut ke dalam kategori:
1. login_issue - Masalah login/akses
2. dpa_issue - Masalah terkait DPA
3. laporan_issue - Masalah laporan/pelaporan
4. technical_issue - Masalah teknis umum
5. general_inquiry - Pertanyaan umum
6. complaint - Keluhan
7. feature_request - Permintaan fitur

Pesan: {message}
Intent:
""",
    
    "sentiment_analysis_prompt": """
Analisis sentimen dari pesan berikut dan berikan skor dari -1 (sangat negatif) hingga 1 (sangat positif):

Pesan: {message}

Sentimen (berikan hanya angka):
""",
    
    "escalation_prompt": """
Berdasarkan percakapan berikut, tentukan apakah perlu eskalasi ke agen manusia:

Percakapan:
{conversation_history}

Pesan terakhir: {last_message}
Sentimen: {sentiment}

Perlu eskalasi? (ya/tidak):
"""
}

# Response templates for common scenarios
RESPONSE_TEMPLATES = {
    "greeting": [
        "Halo! Saya asisten AI untuk sistem SIPD. Ada yang bisa saya bantu hari ini?",
        "Selamat datang di layanan bantuan SIPD! Bagaimana saya bisa membantu Anda?",
        "Hai! Saya siap membantu Anda dengan masalah SIPD. Silakan ceritakan kendala yang Anda hadapi."
    ],
    
    "login_help": [
        "Untuk masalah login, silakan coba langkah berikut:\n1. Pastikan username dan password benar\n2. Periksa koneksi internet\n3. Hapus cache browser\n4. Coba browser lain",
        "Jika tidak bisa login, periksa:\n• Username dan password (case sensitive)\n• Koneksi internet stabil\n• Browser sudah update\n• Tidak ada typo saat input"
    ],
    
    "dpa_help": [
        "Untuk masalah DPA, pastikan:\n1. File format Excel (.xlsx)\n2. Ukuran file maksimal 10MB\n3. Template sesuai standar\n4. Data sudah lengkap dan valid",
        "Kendala upload DPA biasanya karena:\n• Format file salah\n• Ukuran terlalu besar\n• Template tidak sesuai\n• Koneksi internet tidak stabil"
    ],
    
    "technical_help": [
        "Untuk masalah teknis, coba:\n1. Refresh halaman (F5)\n2. Hapus cache browser\n3. Restart browser\n4. Periksa koneksi internet",
        "Solusi umum masalah teknis:\n• Clear browser cache\n• Update browser\n• Disable extensions\n• Coba incognito mode"
    ],
    
    "escalation": [
        "Saya akan menghubungkan Anda dengan tim support untuk bantuan lebih lanjut. Mohon tunggu sebentar.",
        "Untuk masalah ini, saya akan eskalasi ke tim teknis. Mereka akan segera menghubungi Anda.",
        "Tim support akan membantu menyelesaikan masalah Anda. Terima kasih atas kesabaran Anda."
    ],
    
    "no_solution": [
        "Maaf, saya belum menemukan solusi yang tepat. Silakan hubungi tim support di [kontak] untuk bantuan lebih lanjut.",
        "Untuk masalah spesifik ini, saya sarankan menghubungi tim teknis langsung. Mereka akan memberikan solusi yang lebih tepat."
    ]
}

# Suggestion templates based on user intent
SUGGESTION_TEMPLATES = {
    "login_issue": [
        "Reset password",
        "Panduan login",
        "Kontak admin",
        "Cek status sistem"
    ],
    
    "dpa_issue": [
        "Download template DPA",
        "Panduan upload DPA",
        "Validasi data DPA",
        "Kontak tim DPA"
    ],
    
    "laporan_issue": [
        "Format laporan",
        "Jadwal pelaporan",
        "Validasi laporan",
        "Kontak tim laporan"
    ],
    
    "general_inquiry": [
        "FAQ SIPD",
        "Panduan pengguna",
        "Video tutorial",
        "Kontak support"
    ]
}

# Performance monitoring configuration
PERFORMANCE_METRICS = {
    "response_time_threshold": 3.0,  # seconds
    "embedding_time_threshold": 2.0,  # seconds
    "search_time_threshold": 1.0,  # seconds
    "memory_usage_threshold": 512,  # MB
    "cache_hit_rate_target": 0.8,  # 80%
    "error_rate_threshold": 0.05  # 5%
}

# Cost optimization settings
COST_OPTIMIZATION = {
    "max_daily_api_calls": 10000,
    "max_monthly_tokens": 1000000,
    "enable_request_batching": True,
    "cache_expensive_operations": True,
    "use_cheaper_models_for_classification": True,
    "compress_embeddings": True
}

# Security settings
SECURITY_SETTINGS = {
    "max_message_length": 2000,
    "rate_limit_per_minute": 60,
    "rate_limit_per_hour": 1000,
    "enable_input_sanitization": True,
    "log_sensitive_operations": True,
    "encrypt_stored_conversations": True
}

# Initialize enhanced settings
enhanced_settings = NebiusEmbeddingSettings()
chatbot_settings = ChatbotOptimizationSettings()

# Export all settings
__all__ = [
    'NebiusEmbeddingSettings',
    'ChatbotOptimizationSettings', 
    'SIPD_PROMPTS',
    'RESPONSE_TEMPLATES',
    'SUGGESTION_TEMPLATES',
    'PERFORMANCE_METRICS',
    'COST_OPTIMIZATION',
    'SECURITY_SETTINGS',
    'enhanced_settings',
    'chatbot_settings'
]