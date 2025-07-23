# Konfigurasi Khusus untuk Nebius Chatbot
# File ini berisi pengaturan spesifik untuk chatbot yang terhubung dengan Nebius AI

import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dataclasses import dataclass

class NebiusChatbotConfig(BaseSettings):
    """Konfigurasi untuk Nebius Chatbot"""
    
    # Nebius AI Configuration
    nebius_api_key: str = Field(..., env="NEBIUS_API_KEY")
    nebius_base_url: str = Field("https://api.studio.nebius.ai/v1", env="NEBIUS_BASE_URL")
    nebius_model_id: str = Field("meta-llama/Meta-Llama-3.1-70B-Instruct", env="NEBIUS_MODEL_ID")
    nebius_embedding_model: str = Field("BAAI/bge-m3", env="NEBIUS_EMBEDDING_MODEL")
    
    # Chat Configuration
    max_tokens: int = Field(500, env="MAX_TOKENS")
    temperature: float = Field(0.7, env="TEMPERATURE")
    max_conversation_history: int = Field(50, env="MAX_CONVERSATION_HISTORY")
    context_window_size: int = Field(5, env="CONTEXT_WINDOW_SIZE")
    
    # RAG Configuration
    rag_enabled: bool = Field(True, env="RAG_ENABLED")
    rag_similarity_threshold: float = Field(0.7, env="RAG_SIMILARITY_THRESHOLD")
    rag_max_context_length: int = Field(1000, env="RAG_MAX_CONTEXT_LENGTH")
    rag_top_k_results: int = Field(5, env="RAG_TOP_K_RESULTS")
    
    # Vector Store Configuration
    vector_store_path: str = Field("./vector_store", env="VECTOR_STORE_PATH")
    collection_name: str = Field("sipd_knowledge_base", env="COLLECTION_NAME")
    embedding_dimension: int = Field(1536, env="EMBEDDING_DIMENSION")
    
    # Performance Configuration
    enable_caching: bool = Field(True, env="ENABLE_CACHING")
    cache_ttl: int = Field(3600, env="CACHE_TTL")  # 1 hour
    max_concurrent_requests: int = Field(10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(30, env="REQUEST_TIMEOUT")
    
    # Security Configuration
    enable_rate_limiting: bool = Field(True, env="ENABLE_RATE_LIMITING")
    rate_limit_per_minute: int = Field(60, env="RATE_LIMIT_PER_MINUTE")
    max_message_length: int = Field(2000, env="MAX_MESSAGE_LENGTH")
    enable_input_sanitization: bool = Field(True, env="ENABLE_INPUT_SANITIZATION")
    
    # Escalation Configuration
    auto_escalation_enabled: bool = Field(True, env="AUTO_ESCALATION_ENABLED")
    escalation_sentiment_threshold: float = Field(-0.5, env="ESCALATION_SENTIMENT_THRESHOLD")
    escalation_repeated_issues: int = Field(3, env="ESCALATION_REPEATED_ISSUES")
    escalation_webhook_url: str = Field("", env="ESCALATION_WEBHOOK_URL")
    
    # Monitoring Configuration
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    metrics_port: int = Field(8001, env="METRICS_PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # SIPD Specific Configuration
    sipd_base_url: str = Field("", env="SIPD_BASE_URL")
    sipd_admin_contact: str = Field("admin@sipd.go.id", env="SIPD_ADMIN_CONTACT")
    sipd_help_desk_phone: str = Field("+62-21-1234567", env="SIPD_HELP_DESK_PHONE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@dataclass
class ChatbotPersonality:
    """Konfigurasi kepribadian chatbot"""
    
    name: str = "SIPD Assistant"
    greeting_message: str = "Selamat datang di SIPD Chatbot! Saya siap membantu Anda dengan masalah SIPD."
    
    # Response styles berdasarkan situasi
    response_styles: Dict[str, str] = None
    
    def __post_init__(self):
        if self.response_styles is None:
            self.response_styles = {
                "formal": "Menggunakan bahasa formal dan profesional",
                "friendly": "Menggunakan bahasa ramah dan santai",
                "empathetic": "Menunjukkan empati dan pengertian",
                "technical": "Memberikan penjelasan teknis yang detail",
                "urgent": "Merespons dengan prioritas tinggi dan cepat"
            }

@dataclass
class SIPDKnowledgeBase:
    """Knowledge base untuk informasi SIPD"""
    
    # FAQ Categories
    faq_categories: List[str] = None
    
    # Common issues dan solutions
    common_solutions: Dict[str, List[str]] = None
    
    # Contact information
    support_contacts: Dict[str, str] = None
    
    def __post_init__(self):
        if self.faq_categories is None:
            self.faq_categories = [
                "Login dan Akses",
                "DPA (Dokumen Pelaksanaan Anggaran)",
                "Laporan dan Pelaporan",
                "Masalah Teknis",
                "Panduan Penggunaan",
                "Backup dan Recovery",
                "Keamanan Data"
            ]
        
        if self.common_solutions is None:
            self.common_solutions = {
                "login_issue": [
                    "Pastikan username dan password benar",
                    "Coba reset password melalui menu 'Lupa Password'",
                    "Hapus cache dan cookies browser",
                    "Gunakan browser yang didukung (Chrome, Firefox, Edge)",
                    "Periksa koneksi internet",
                    "Hubungi admin jika masalah berlanjut"
                ],
                "dpa_issue": [
                    "Periksa format file (harus .xlsx atau .xls)",
                    "Pastikan semua kolom wajib terisi",
                    "Validasi data sesuai template terbaru",
                    "Cek ukuran file (maksimal 10MB)",
                    "Pastikan koneksi stabil saat upload",
                    "Download template DPA terbaru dari sistem"
                ],
                "laporan_issue": [
                    "Refresh halaman laporan",
                    "Periksa filter tanggal yang dipilih",
                    "Pastikan data sudah tersinkronisasi",
                    "Coba export dalam format berbeda",
                    "Gunakan browser yang didukung",
                    "Tunggu beberapa saat jika server sedang sibuk"
                ],
                "technical_issue": [
                    "Restart browser dan coba lagi",
                    "Clear cache dan cookies",
                    "Disable browser extensions",
                    "Coba dari komputer/jaringan lain",
                    "Periksa firewall dan antivirus",
                    "Laporkan ke tim IT dengan screenshot error"
                ]
            }
        
        if self.support_contacts is None:
            self.support_contacts = {
                "email": "support@sipd.go.id",
                "phone": "+62-21-1234567",
                "whatsapp": "+62-812-3456789",
                "telegram": "@sipd_support",
                "website": "https://sipd.kemendagri.go.id",
                "help_desk": "https://help.sipd.kemendagri.go.id"
            }

class PromptTemplates:
    """Template prompts untuk berbagai skenario"""
    
    SYSTEM_PROMPT_BASE = """
Anda adalah {bot_name}, asisten AI untuk Sistem Informasi Pemerintah Daerah (SIPD).

Tugas Anda:
1. Membantu pengguna dengan masalah SIPD
2. Memberikan solusi yang akurat dan praktis
3. Menunjukkan empati dan profesionalisme
4. Mengarahkan ke sumber daya yang tepat

Guidelines:
- Gunakan bahasa Indonesia yang jelas dan profesional
- Berikan langkah-langkah yang spesifik dan mudah diikuti
- Jika tidak yakin, arahkan ke admin atau dokumentasi resmi
- Selalu konfirmasi pemahaman user sebelum memberikan solusi kompleks
- Tunjukkan empati terhadap frustrasi user

Kontak Support:
- Email: {support_email}
- Phone: {support_phone}
- Website: {support_website}
"""
    
    INTENT_CLASSIFICATION_PROMPT = """
Analisis pesan user berikut dan klasifikasikan intent-nya:

Pesan: "{message}"

Kategori yang tersedia:
- login_issue: Masalah login, akses, atau autentikasi
- dpa_issue: Masalah DPA (upload, validasi, persetujuan)
- laporan_issue: Masalah laporan (akses, export, tampilan)
- technical_issue: Masalah teknis sistem atau error
- general_inquiry: Pertanyaan umum tentang SIPD
- complaint: Keluhan atau kritik
- praise: Pujian atau feedback positif
- help_request: Permintaan bantuan umum

Jawab hanya dengan nama kategori yang paling sesuai.
"""
    
    SENTIMENT_ANALYSIS_PROMPT = """
Analisis sentiment dari pesan berikut:

Pesan: "{message}"

Klasifikasi sentiment:
- positive: Sentiment positif (puas, senang, terima kasih)
- neutral: Sentiment netral (pertanyaan biasa, informasi)
- negative: Sentiment negatif (frustrasi, marah, kecewa)

Jawab hanya dengan nama sentiment.
"""
    
    RESPONSE_GENERATION_PROMPT = """
{system_prompt}

Konteks Percakapan:
Intent: {intent}
Sentiment: {sentiment}

{conversation_history}

{rag_context}

Pesan User: "{user_message}"

Instruksi Response:
1. Berikan jawaban yang sesuai dengan intent dan sentiment
2. Gunakan informasi dari knowledge base jika tersedia
3. Berikan solusi praktis dan langkah-langkah yang jelas
4. Tunjukkan empati jika user mengalami masalah
5. Sertakan informasi kontak jika diperlukan eskalasi

Response:
"""
    
    ESCALATION_PROMPT = """
Berdasarkan analisis percakapan, tentukan apakah situasi ini memerlukan eskalasi ke human agent.

Faktor yang perlu dipertimbangkan:
- Sentiment: {sentiment}
- Intent: {intent}
- Jumlah percakapan: {conversation_count}
- Masalah berulang: {repeated_issues}
- Kompleksitas masalah: {complexity}

Kriteria eskalasi:
- Sentiment sangat negatif
- Masalah teknis kompleks yang tidak bisa diselesaikan
- User meminta bicara dengan manusia
- Keluhan formal atau kritik serius
- Masalah berulang tanpa solusi

Jawab dengan 'ya' jika perlu eskalasi, 'tidak' jika tidak perlu.
"""

# Global configuration instance
config = NebiusChatbotConfig()
personality = ChatbotPersonality()
sipd_kb = SIPDKnowledgeBase()
prompt_templates = PromptTemplates()

# Helper functions
def get_system_prompt(bot_name: str = None, support_contacts: Dict[str, str] = None) -> str:
    """Generate system prompt dengan konfigurasi yang sesuai"""
    if bot_name is None:
        bot_name = personality.name
    
    if support_contacts is None:
        support_contacts = sipd_kb.support_contacts
    
    return prompt_templates.SYSTEM_PROMPT_BASE.format(
        bot_name=bot_name,
        support_email=support_contacts.get('email', ''),
        support_phone=support_contacts.get('phone', ''),
        support_website=support_contacts.get('website', '')
    )

def get_response_style(intent: str, sentiment: str) -> str:
    """Tentukan style response berdasarkan intent dan sentiment"""
    if sentiment == 'negative':
        if intent in ['complaint', 'technical_issue']:
            return 'empathetic'
        else:
            return 'friendly'
    elif intent in ['technical_issue', 'dpa_issue']:
        return 'technical'
    elif sentiment == 'positive':
        return 'friendly'
    else:
        return 'formal'

def get_common_solutions(intent: str) -> List[str]:
    """Ambil solusi umum berdasarkan intent"""
    return sipd_kb.common_solutions.get(intent, [])

def should_include_contact_info(intent: str, sentiment: str) -> bool:
    """Tentukan apakah perlu menyertakan informasi kontak"""
    return (
        sentiment == 'negative' or 
        intent in ['complaint', 'technical_issue'] or
        intent == 'help_request'
    )

# Validation functions
def validate_config() -> bool:
    """Validasi konfigurasi chatbot"""
    try:
        # Check required environment variables
        required_vars = ['NEBIUS_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"Missing required environment variables: {missing_vars}")
            return False
        
        # Validate configuration values
        if config.max_tokens <= 0:
            print("max_tokens must be positive")
            return False
        
        if not (0.0 <= config.temperature <= 2.0):
            print("temperature must be between 0.0 and 2.0")
            return False
        
        if config.rag_similarity_threshold < 0 or config.rag_similarity_threshold > 1:
            print("rag_similarity_threshold must be between 0 and 1")
            return False
        
        return True
        
    except Exception as e:
        print(f"Configuration validation error: {e}")
        return False

if __name__ == "__main__":
    # Test configuration
    if validate_config():
        print("✅ Configuration is valid")
        print(f"Bot Name: {personality.name}")
        print(f"Model: {config.nebius_model_id}")
        print(f"RAG Enabled: {config.rag_enabled}")
        print(f"Caching Enabled: {config.enable_caching}")
    else:
        print("❌ Configuration validation failed")