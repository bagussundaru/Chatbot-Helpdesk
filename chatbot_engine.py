from typing import List, Dict, Any, Optional
from loguru import logger
from rag_system import SIPDRAGSystem
from nebius_client import NebiusAIClient
from config import settings
import time
import json

class SIPDChatbotEngine:
    """Core engine untuk SIPD AI Chatbot yang menggabungkan RAG dan Nebius AI"""
    
    def __init__(self):
        self.rag_system = SIPDRAGSystem()
        self.ai_client = NebiusAIClient()
        self.conversation_history = {}
        
        # Initialize knowledge base
        self.rag_system.initialize_knowledge_base()
        
        logger.info("SIPD Chatbot Engine initialized")
    
    def process_message(
        self, 
        user_message: str, 
        session_id: str = "default",
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process user message and generate response"""
        
        start_time = time.time()
        
        try:
            # Initialize session if not exists
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            
            # Analyze user message
            intent = self.ai_client.classify_intent(user_message)
            sentiment = self.ai_client.analyze_sentiment(user_message)
            
            # Get relevant context from RAG system
            rag_context = self.rag_system.get_context_for_query(user_message)
            
            # Prepare conversation context
            conversation_context = self._prepare_conversation_context(
                session_id, user_message, intent, sentiment, rag_context
            )
            
            # Generate response using Nebius AI
            ai_response = self.ai_client.generate_response(
                messages=conversation_context,
                context=rag_context
            )
            
            # Update conversation history
            self._update_conversation_history(
                session_id, user_message, ai_response
            )
            
            # Prepare response
            response = {
                "response": ai_response,
                "intent": intent,
                "sentiment": sentiment,
                "session_id": session_id,
                "processing_time": round(time.time() - start_time, 2),
                "has_context": bool(rag_context and rag_context != "Maaf, saya tidak menemukan informasi yang relevan dalam database."),
                "suggestions": self._generate_suggestions(intent, sentiment)
            }
            
            logger.info(f"Processed message for session {session_id} in {response['processing_time']}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": "Maaf, terjadi kesalahan dalam memproses pesan Anda. Silakan coba lagi atau hubungi tim teknis.",
                "error": str(e),
                "session_id": session_id,
                "processing_time": round(time.time() - start_time, 2)
            }
    
    def _prepare_conversation_context(
        self, 
        session_id: str, 
        user_message: str, 
        intent: Dict[str, Any], 
        sentiment: Dict[str, Any],
        rag_context: str
    ) -> List[Dict[str, str]]:
        """Prepare conversation context for AI model"""
        
        # Get recent conversation history
        history = self.conversation_history.get(session_id, [])
        recent_history = history[-settings.max_conversation_history:]
        
        # Build context messages
        context_messages = []
        
        # Add conversation history
        for msg in recent_history:
            context_messages.extend([
                {"role": "user", "content": msg["user"]},
                {"role": "assistant", "content": msg["assistant"]}
            ])
        
        # Add current user message with enhanced context
        enhanced_message = user_message
        
        # Add intent information if available
        if intent.get("menu") != "Lainnya":
            enhanced_message += f" [Menu: {intent['menu']}]"
        
        # Add empathy cue if needed
        if sentiment.get("needs_empathy"):
            enhanced_message += f" [Pengguna tampak {sentiment.get('emotion', 'frustrasi')}]"
        
        context_messages.append({
            "role": "user", 
            "content": enhanced_message
        })
        
        return context_messages
    
    def _update_conversation_history(
        self, 
        session_id: str, 
        user_message: str, 
        ai_response: str
    ):
        """Update conversation history for session"""
        
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        self.conversation_history[session_id].append({
            "user": user_message,
            "assistant": ai_response,
            "timestamp": time.time()
        })
        
        # Keep only recent history
        if len(self.conversation_history[session_id]) > settings.max_conversation_history:
            self.conversation_history[session_id] = self.conversation_history[session_id][-settings.max_conversation_history:]
    
    def _generate_suggestions(
        self, 
        intent: Dict[str, Any], 
        sentiment: Dict[str, Any]
    ) -> List[str]:
        """Generate helpful suggestions based on intent and sentiment"""
        
        suggestions = []
        
        menu = intent.get("menu", "")
        jenis_masalah = intent.get("jenis_masalah", "")
        urgensi = intent.get("urgensi", "")
        
        # Menu-specific suggestions
        if menu == "Login/Akses":
            suggestions.extend([
                "Cek panduan reset password",
                "Verifikasi koneksi internet",
                "Hubungi admin untuk reset akun"
            ])
        elif menu == "Penganggaran":
            suggestions.extend([
                "Lihat tutorial input anggaran",
                "Cek status persetujuan DPA",
                "Panduan revisi anggaran"
            ])
        elif menu == "Pelaporan":
            suggestions.extend([
                "Format laporan yang benar",
                "Jadwal pelaporan bulanan",
                "Cara export laporan"
            ])
        
        # Urgency-based suggestions
        if urgensi == "Tinggi":
            suggestions.insert(0, "Hubungi hotline teknis: 021-XXXXXXX")
        
        # Sentiment-based suggestions
        if sentiment.get("needs_empathy"):
            suggestions.append("Tim kami siap membantu Anda 24/7")
        
        return suggestions[:3]  # Return max 3 suggestions
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        return self.conversation_history.get(session_id, [])
    
    def clear_session_history(self, session_id: str) -> bool:
        """Clear conversation history for a session"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            return True
        return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            rag_stats = self.rag_system.get_collection_stats()
            ai_health = self.ai_client.health_check()
            
            return {
                "rag_system": rag_stats,
                "ai_service_healthy": ai_health,
                "active_sessions": len(self.conversation_history),
                "total_conversations": sum(len(history) for history in self.conversation_history.values())
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}
    
    def add_feedback(self, session_id: str, message_index: int, rating: int, comment: str = ""):
        """Add user feedback for continuous improvement"""
        try:
            # This could be expanded to store feedback in a database
            feedback = {
                "session_id": session_id,
                "message_index": message_index,
                "rating": rating,  # 1-5 scale
                "comment": comment,
                "timestamp": time.time()
            }
            
            # Log feedback for now (could be stored in database)
            logger.info(f"User feedback received: {feedback}")
            
            return True
        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
            return False
    
    def escalate_to_human(self, session_id: str, reason: str = "") -> Dict[str, Any]:
        """Escalate conversation to human agent"""
        try:
            escalation_data = {
                "session_id": session_id,
                "conversation_history": self.get_session_history(session_id),
                "escalation_reason": reason,
                "timestamp": time.time(),
                "ticket_id": f"SIPD-{int(time.time())}"
            }
            
            # Log escalation (in real implementation, this would create a support ticket)
            logger.info(f"Escalation created: {escalation_data['ticket_id']}")
            
            return {
                "success": True,
                "ticket_id": escalation_data["ticket_id"],
                "message": f"Tiket dukungan {escalation_data['ticket_id']} telah dibuat. Tim teknis akan menghubungi Anda segera."
            }
            
        except Exception as e:
            logger.error(f"Error creating escalation: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Maaf, terjadi kesalahan dalam membuat tiket dukungan. Silakan hubungi hotline langsung."
            }

if __name__ == "__main__":
    # Test the chatbot engine
    engine = SIPDChatbotEngine()
    
    # Test conversation
    test_messages = [
        "Halo, saya tidak bisa login ke SIPD",
        "Muncul error 500 ketika saya coba masuk",
        "Sudah coba clear cache tapi masih sama"
    ]
    
    session_id = "test_session"
    
    for message in test_messages:
        print(f"\nUser: {message}")
        response = engine.process_message(message, session_id)
        print(f"Bot: {response['response']}")
        print(f"Intent: {response.get('intent', {})}")
        print(f"Suggestions: {response.get('suggestions', [])}")
    
    # Test system stats
    stats = engine.get_system_stats()
    print(f"\nSystem Stats: {stats}")