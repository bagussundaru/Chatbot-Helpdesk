# Language Detector for SIPD AI Chatbot
# Implementasi deteksi bahasa untuk mendukung fitur multilingual

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
from loguru import logger
import re

# Optional imports for better language detection
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    logger.warning("FastText not available. Using fallback language detection.")

try:
    import langid
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False
    logger.warning("LangID not available. Using fallback language detection.")

class LanguageDetector:
    """Komponen untuk deteksi bahasa"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.supported_languages = [
            "id", # Indonesian
            "en", # English
            "jv", # Javanese
            "su", # Sundanese
            "ms"  # Malay
        ]
        self.language_keywords = {
            "id": [
                "apa", "bagaimana", "tolong", "mohon", "tidak", "bisa", "silakan",
                "terima", "kasih", "selamat", "pagi", "siang", "malam", "saya", "kamu",
                "anda", "mereka", "kami", "kita", "ini", "itu", "dan", "atau", "tapi",
                "namun", "jika", "kalau", "ketika", "karena", "untuk", "dari", "dengan",
                "pada", "dalam", "tentang", "mengenai", "masalah", "problem", "error",
                "gagal", "berhasil", "sukses", "login", "masuk", "keluar", "daftar",
                "laporan", "anggaran", "data", "sistem", "aplikasi", "website", "situs"
            ],
            "en": [
                "what", "how", "please", "help", "can", "could", "thank", "you", "good",
                "morning", "afternoon", "evening", "night", "i", "you", "they", "we", "this",
                "that", "and", "or", "but", "however", "if", "when", "because", "for", "from",
                "with", "on", "in", "about", "regarding", "issue", "problem", "error", "fail",
                "success", "successful", "login", "sign", "out", "register", "report", "budget",
                "data", "system", "application", "website", "site"
            ],
            "jv": [
                "opo", "piye", "tulung", "ora", "iso", "monggo", "matur", "nuwun", "sugeng",
                "enjang", "siang", "sonten", "dalu", "kulo", "sampeyan", "panjenengan", "lan",
                "utowo", "nanging", "menawi", "nalika", "amargi", "kangge", "saking", "kaliyan",
                "wonten", "ing", "bab", "masalah", "gagal", "kasil", "mlebet", "medal", "laporan"
            ],
            "su": [
                "naon", "kumaha", "punten", "henteu", "tiasa", "mangga", "hatur", "nuhun",
                "wilujeng", "enjing", "siang", "sonten", "wengi", "abdi", "anjeun", "sareng",
                "atanapi", "tapi", "lamun", "nalika", "kusabab", "keur", "ti", "jeung", "dina",
                "ngeunaan", "masalah", "gagal", "hasil", "lebet", "kaluar", "laporan"
            ],
            "ms": [
                "apa", "bagaimana", "tolong", "tidak", "boleh", "silakan", "terima", "kasih",
                "selamat", "pagi", "tengahari", "petang", "malam", "saya", "awak", "anda",
                "mereka", "kami", "kita", "ini", "itu", "dan", "atau", "tetapi", "jika", "bila",
                "apabila", "kerana", "untuk", "dari", "dengan", "pada", "dalam", "tentang",
                "mengenai", "masalah", "ralat", "gagal", "berjaya", "log", "masuk", "keluar",
                "laporan", "belanjawan", "data", "sistem", "aplikasi", "laman", "sesawang"
            ]
        }
        
        # Setup logging
        logger.add("logs/language_detector.log", rotation="10 MB", level="INFO")
    
    async def initialize(self) -> bool:
        """Initialize language detector"""
        try:
            # Try to load FastText model if available
            if FASTTEXT_AVAILABLE and self.model_path:
                try:
                    self.model = fasttext.load_model(self.model_path)
                    logger.info(f"FastText language detection model loaded from {self.model_path}")
                    return True
                except Exception as e:
                    logger.error(f"Error loading FastText model: {e}")
            
            # If FastText not available or failed, check if LangID is available
            if LANGID_AVAILABLE:
                logger.info("Using LangID for language detection")
                return True
            
            # Fallback to keyword-based detection
            logger.info("Using keyword-based language detection")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing language detector: {e}")
            return False
    
    async def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        try:
            if not text or len(text.strip()) == 0:
                return "id"  # Default to Indonesian for empty text
            
            # Clean text
            text_clean = self._clean_text(text)
            
            # Try FastText if available
            if FASTTEXT_AVAILABLE and self.model:
                try:
                    prediction = self.model.predict(text_clean)[0][0]
                    lang_code = prediction.replace('__label__', '')
                    
                    # Check if detected language is supported
                    if lang_code in self.supported_languages:
                        logger.info(f"FastText detected language: {lang_code} for text: {text[:50]}...")
                        return lang_code
                except Exception as e:
                    logger.error(f"Error using FastText for detection: {e}")
            
            # Try LangID if available
            if LANGID_AVAILABLE:
                try:
                    lang_code, confidence = langid.classify(text_clean)
                    
                    # Check if detected language is supported and confidence is high enough
                    if lang_code in self.supported_languages and confidence > 0.5:
                        logger.info(f"LangID detected language: {lang_code} (confidence: {confidence:.2f}) for text: {text[:50]}...")
                        return lang_code
                except Exception as e:
                    logger.error(f"Error using LangID for detection: {e}")
            
            # Fallback to keyword-based detection
            return self._keyword_based_detection(text_clean)
            
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "id"  # Default to Indonesian on error
    
    def _clean_text(self, text: str) -> str:
        """Clean text for language detection"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces and alphanumeric
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _keyword_based_detection(self, text: str) -> str:
        """Detect language based on keyword matching"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Count matches for each language
        lang_scores = {lang: 0 for lang in self.supported_languages}
        
        for word in words:
            for lang, keywords in self.language_keywords.items():
                if word in keywords:
                    lang_scores[lang] += 1
        
        # Find language with highest score
        max_score = 0
        detected_lang = "id"  # Default to Indonesian
        
        for lang, score in lang_scores.items():
            if score > max_score:
                max_score = score
                detected_lang = lang
        
        logger.info(f"Keyword-based detection: {detected_lang} (score: {max_score}) for text: {text[:50]}...")
        return detected_lang
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages"""
        language_names = {
            "id": "Indonesian",
            "en": "English",
            "jv": "Javanese",
            "su": "Sundanese",
            "ms": "Malay"
        }
        
        return [
            {"code": code, "name": language_names.get(code, code)}
            for code in self.supported_languages
        ]

# Example usage
if __name__ == "__main__":
    async def test_language_detector():
        # Initialize detector
        detector = LanguageDetector()
        await detector.initialize()
        
        # Test texts
        test_texts = {
            "id": "Selamat pagi, saya tidak bisa login ke sistem SIPD. Tolong bantu saya.",
            "en": "Good morning, I cannot login to the SIPD system. Please help me.",
            "jv": "Sugeng enjang, kulo mboten saged mlebet wonten sistem SIPD. Tulung bantu kulo.",
            "ms": "Selamat pagi, saya tidak boleh log masuk ke sistem SIPD. Tolong bantu saya.",
            "mixed": "Hello, saya ada masalah dengan login SIPD. Can you help me please?"
        }
        
        print("\nLanguage detection results:")
        for label, text in test_texts.items():
            detected = await detector.detect_language(text)
            print(f"Text ({label}): {text}")
            print(f"Detected language: {detected}\n")
        
        # Get supported languages
        supported = detector.get_supported_languages()
        
        print("\nSupported languages:")
        for lang in supported:
            print(f"{lang['code']}: {lang['name']}")
    
    # Run the test
    asyncio.run(test_language_detector())