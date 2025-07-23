import openai
from typing import List, Dict, Any, Optional
from loguru import logger
from config import settings
import json
import time

class NebiusAIClient:
    """Client untuk berinteraksi dengan Nebius AI Studio"""
    
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=settings.nebius_api_key,
            base_url=settings.nebius_base_url
        )
        self.model_id = settings.nebius_model_id
        
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        context: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate response using Nebius AI model"""
        
        if temperature is None:
            temperature = settings.temperature
        if max_tokens is None:
            max_tokens = settings.max_tokens
            
        try:
            # Prepare system message with context if provided
            system_message = self._create_system_message(context)
            
            # Prepare full message list
            full_messages = [system_message] + messages
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=settings.response_timeout
            )
            
            # Extract response content
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content.strip()
            else:
                logger.error("No response content received from Nebius AI")
                return "Maaf, terjadi kesalahan dalam memproses permintaan Anda."
                
        except openai.APITimeoutError:
            logger.error("Timeout when calling Nebius AI API")
            return "Maaf, respons memakan waktu terlalu lama. Silakan coba lagi."
        except openai.APIError as e:
            logger.error(f"Nebius AI API error: {e}")
            return "Maaf, terjadi kesalahan pada layanan AI. Silakan coba lagi nanti."
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {e}")
            return "Maaf, terjadi kesalahan yang tidak terduga. Silakan coba lagi."
    
    def _create_system_message(self, context: Optional[str] = None) -> Dict[str, str]:
        """Create system message with SIPD chatbot persona and context"""
        
        base_prompt = """Anda adalah asisten virtual SIPD (Sistem Informasi Pemerintahan Daerah) yang cerdas dan membantu. 

Karakteristik Anda:
- Ramah, profesional, dan empatik
- Memahami terminologi dan proses SIPD
- Memberikan solusi yang jelas dan langkah demi langkah
- Mengakui keterbatasan dan menawarkan eskalasi jika diperlukan
- Menggunakan bahasa Indonesia yang sopan dan mudah dipahami

Tugas Anda:
- Membantu pengguna menyelesaikan masalah teknis SIPD
- Memberikan panduan berdasarkan data historis aduan
- Mengklasifikasi masalah berdasarkan menu/modul SIPD
- Memberikan solusi alternatif jika memungkinkan

Jika Anda tidak yakin dengan jawaban, katakan dengan jujur dan tawarkan untuk menghubungkan dengan tim teknis."""
        
        if context:
            base_prompt += f"\n\nInformasi relevan dari database pengetahuan:\n{context}"
            base_prompt += "\n\nGunakan informasi di atas untuk memberikan jawaban yang akurat dan spesifik."
        
        return {
            "role": "system",
            "content": base_prompt
        }
    
    def classify_intent(self, user_message: str) -> Dict[str, Any]:
        """Classify user intent and extract key information"""
        
        classification_prompt = """Analisis pesan pengguna dan klasifikasikan berdasarkan:

1. MENU/MODUL SIPD:
   - Penganggaran
   - Penatausahaan  
   - Akuntansi
   - Pelaporan
   - Login/Akses
   - Lainnya

2. JENIS MASALAH:
   - Error/Bug
   - Panduan/Tutorial
   - Akses/Login
   - Data/Input
   - Laporan
   - Lainnya

3. TINGKAT URGENSI:
   - Tinggi (sistem down, tidak bisa akses)
   - Sedang (fitur tidak berfungsi)
   - Rendah (pertanyaan umum)

Berikan hasil dalam format JSON dengan key: menu, jenis_masalah, urgensi, dan ringkasan_masalah."""
        
        try:
            messages = [
                {"role": "system", "content": classification_prompt},
                {"role": "user", "content": user_message}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=200
            )
            
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content.strip()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Fallback classification
                    return {
                        "menu": "Lainnya",
                        "jenis_masalah": "Lainnya",
                        "urgensi": "Sedang",
                        "ringkasan_masalah": user_message[:100]
                    }
            
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            
        # Default classification
        return {
            "menu": "Lainnya",
            "jenis_masalah": "Lainnya", 
            "urgensi": "Sedang",
            "ringkasan_masalah": user_message[:100]
        }
    
    def analyze_sentiment(self, user_message: str) -> Dict[str, Any]:
        """Analyze sentiment of user message"""
        
        sentiment_prompt = """Analisis sentimen dari pesan pengguna. Berikan hasil dalam format JSON dengan:
- sentiment: 'positif', 'negatif', atau 'netral'
- confidence: skor 0-1
- emotion: 'senang', 'frustrasi', 'bingung', 'marah', 'netral'
- needs_empathy: true/false (apakah perlu respons empatik)"""
        
        try:
            messages = [
                {"role": "system", "content": sentiment_prompt},
                {"role": "user", "content": user_message}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.1,
                max_tokens=150
            )
            
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content.strip()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            
        # Default sentiment
        return {
            "sentiment": "netral",
            "confidence": 0.5,
            "emotion": "netral",
            "needs_empathy": False
        }
    
    def health_check(self) -> bool:
        """Check if Nebius AI service is available"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10,
                timeout=10
            )
            return True
        except Exception as e:
            logger.error(f"Nebius AI health check failed: {e}")
            return False

if __name__ == "__main__":
    # Test the Nebius client
    client = NebiusAIClient()
    
    # Test health check
    if client.health_check():
        print("Nebius AI service is available")
        
        # Test classification
        test_message = "Saya tidak bisa login ke SIPD, muncul error 500"
        intent = client.classify_intent(test_message)
        print(f"Intent classification: {intent}")
        
        # Test sentiment
        sentiment = client.analyze_sentiment(test_message)
        print(f"Sentiment analysis: {sentiment}")
    else:
        print("Nebius AI service is not available")