# Meta-LLaMA-3.1-70B-Instruct Client for SIPD AI Chatbot
# Implementasi client untuk model Meta-LLaMA-3.1-70B-Instruct

import os
import json
import asyncio
import httpx
from typing import List, Dict, Any, Optional, Union
from loguru import logger
from datetime import datetime
from config import settings

class MetaLLMClient:
    """Client for Meta-LLaMA-3.1-70B-Instruct model"""
    
    def __init__(self):
        self.api_key = os.getenv("META_LLM_API_KEY", "")
        self.base_url = os.getenv("META_LLM_BASE_URL", "https://api.llama.ai/v1")
        self.model_id = "meta-llama-3.1-70b-instruct"
        self.max_retries = 3
        self.timeout = 60
        self.client = None
        self.initialized = False
        
        # Setup logging
        logger.add("logs/meta_llm_client.log", rotation="10 MB", level="INFO")
    
    async def initialize(self) -> bool:
        """Initialize LLM client"""
        try:
            # Create async client
            self.client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            # Test connection
            if self.api_key:
                # In a real implementation, we would test the connection here
                # For now, we'll just log a message
                logger.info(f"Meta LLM client initialized with model: {self.model_id}")
                self.initialized = True
                return True
            else:
                logger.warning("No API key provided for Meta LLM client. Running in simulation mode.")
                self.initialized = True
                return True
                
        except Exception as e:
            logger.error(f"Error initializing Meta LLM client: {e}")
            return False
    
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop_sequences: List[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate response from Meta LLM"""
        try:
            # Check if client is initialized
            if not self.initialized:
                await self.initialize()
            
            # In a real implementation, we would call the Meta LLM API here
            # For now, we'll simulate a response
            
            # If API key is provided, attempt to call the API
            if self.api_key and self.client:
                try:
                    # Build request payload
                    payload = {
                        "model": self.model_id,
                        "messages": [
                            {"role": "system", "content": system_prompt or "You are a helpful assistant."}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "stream": stream
                    }
                    
                    # Add user message
                    payload["messages"].append({"role": "user", "content": prompt})
                    
                    # Add stop sequences if provided
                    if stop_sequences:
                        payload["stop"] = stop_sequences
                    
                    # Make API call
                    response = await self.client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload
                    )
                    
                    # Check response
                    if response.status_code == 200:
                        result = response.json()
                        return {
                            "response": result["choices"][0]["message"]["content"],
                            "tokens_used": result.get("usage", {}).get("total_tokens", 0),
                            "model": self.model_id,
                            "finish_reason": result["choices"][0].get("finish_reason", "stop")
                        }
                    else:
                        logger.error(f"API error: {response.status_code} - {response.text}")
                        # Fall back to simulation
                        return self._simulate_response(prompt)
                        
                except Exception as e:
                    logger.error(f"Error calling Meta LLM API: {e}")
                    # Fall back to simulation
                    return self._simulate_response(prompt)
            else:
                # Simulate response
                return self._simulate_response(prompt)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "Maaf, saya mengalami kesulitan memproses permintaan Anda. Silakan coba lagi.",
                "error": str(e)
            }
    
    def _simulate_response(self, prompt: str) -> Dict[str, Any]:
        """Simulate a response for testing purposes"""
        # Simulate API call delay
        asyncio.sleep(1)
        
        # Simple response generation based on prompt keywords
        prompt_lower = prompt.lower()
        
        if "login" in prompt_lower or "masuk" in prompt_lower:
            response = "Untuk masalah login SIPD, silakan coba langkah berikut:\n1. Pastikan username dan password benar\n2. Clear cache browser\n3. Coba browser lain\n4. Hubungi admin jika masih bermasalah"
        elif "dpa" in prompt_lower or "anggaran" in prompt_lower:
            response = "Untuk masalah DPA, pastikan:\n1. Semua field mandatory terisi\n2. Format data sesuai\n3. Koneksi internet stabil\n4. Refresh halaman dan coba lagi"
        elif "laporan" in prompt_lower or "report" in prompt_lower:
            response = "Untuk masalah laporan:\n1. Cek format data\n2. Pastikan periode laporan benar\n3. Coba export dengan data lebih sedikit\n4. Hubungi tim teknis jika error berlanjut"
        elif "terima kasih" in prompt_lower or "thank" in prompt_lower:
            response = "Sama-sama! Senang bisa membantu Anda. Jika ada pertanyaan lain, jangan ragu untuk bertanya kembali."
        else:
            response = "Terima kasih atas pertanyaan Anda. Saya adalah asisten virtual SIPD yang siap membantu menyelesaikan masalah teknis. Bisa Anda jelaskan lebih detail masalah yang Anda hadapi?"
        
        return {
            "response": response,
            "tokens_used": len(prompt.split()) + len(response.split()),
            "model": f"{self.model_id} (simulated)",
            "finish_reason": "stop"
        }
    
    async def close(self):
        """Close the client"""
        if self.client:
            await self.client.aclose()
            logger.info("Meta LLM client closed")

# Example usage
if __name__ == "__main__":
    async def test_client():
        client = MetaLLMClient()
        await client.initialize()
        
        # Test with a simple prompt
        response = await client.generate_response(
            prompt="Bagaimana cara mengatasi masalah login SIPD?",
            system_prompt="Anda adalah asisten AI untuk Sistem Informasi Pemerintah Daerah (SIPD)."
        )
        
        print("\nResponse:")
        print(response["response"])
        print(f"\nModel: {response['model']}")
        print(f"Tokens used: {response['tokens_used']}")
        print(f"Finish reason: {response['finish_reason']}")
        
        await client.close()
    
    # Run the test
    asyncio.run(test_client())