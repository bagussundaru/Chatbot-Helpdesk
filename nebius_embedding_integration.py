# Nebius Embedding Integration for SIPD Chatbot
# Enhanced RAG System with Nebius AI Studio Embeddings

import os
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import httpx
from loguru import logger
from config import settings

@dataclass
class EmbeddingResult:
    """Result from embedding API call"""
    embedding: List[float]
    tokens_used: int
    model: str
    
class NebiusEmbeddingClient:
    """Client for Nebius AI Studio Embedding API"""
    
    def __init__(self):
        self.api_key = settings.nebius_api_key
        self.base_url = settings.nebius_base_url
        self.embedding_model = getattr(settings, 'nebius_embedding_model', 'text-embedding-ada-002')
        self.max_retries = 3
        self.timeout = 30
        
    async def get_embedding(self, text: str) -> Optional[EmbeddingResult]:
        """Get embedding for a single text"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.embedding_model,
                    "input": text,
                    "encoding_format": "float"
                }
                
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    embedding_data = data['data'][0]
                    
                    return EmbeddingResult(
                        embedding=embedding_data['embedding'],
                        tokens_used=data['usage']['total_tokens'],
                        model=self.embedding_model
                    )
                else:
                    logger.error(f"Embedding API error: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    async def get_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[Optional[EmbeddingResult]]:
        """Get embeddings for multiple texts in batches"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self.get_embedding(text) for text in batch],
                return_exceptions=True
            )
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch embedding error: {result}")
                    results.append(None)
                else:
                    results.append(result)
            
            # Rate limiting - wait between batches
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return results

class EnhancedRAGSystem:
    """Enhanced RAG System using Nebius embeddings"""
    
    def __init__(self, collection_name: str = "sipd_knowledge_base"):
        self.collection_name = collection_name
        self.embedding_client = NebiusEmbeddingClient()
        self.vector_store = None
        self.embedding_cache = {}  # Simple cache for embeddings
        
    async def initialize_vector_store(self):
        """Initialize ChromaDB with Nebius embeddings"""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            # Initialize ChromaDB client
            chroma_client = chromadb.PersistentClient(
                path=settings.vector_db_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection with custom embedding function
            self.vector_store = chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._nebius_embedding_function,
                metadata={"description": "SIPD Knowledge Base with Nebius Embeddings"}
            )
            
            logger.info(f"Vector store initialized with collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            return False
    
    def _nebius_embedding_function(self, texts: List[str]) -> List[List[float]]:
        """Custom embedding function for ChromaDB using Nebius"""
        try:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            embeddings_results = loop.run_until_complete(
                self.embedding_client.get_embeddings_batch(texts)
            )
            
            embeddings = []
            for result in embeddings_results:
                if result and result.embedding:
                    embeddings.append(result.embedding)
                else:
                    # Fallback to zero vector if embedding fails
                    embeddings.append([0.0] * 1536)  # Default embedding size
            
            loop.close()
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in embedding function: {e}")
            # Return zero vectors as fallback
            return [[0.0] * 1536 for _ in texts]
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to vector store with Nebius embeddings"""
        try:
            if not self.vector_store:
                await self.initialize_vector_store()
            
            # Prepare document texts and metadata
            texts = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                # Combine relevant fields for embedding
                text_parts = []
                if doc.get('MENU'):
                    text_parts.append(f"Menu: {doc['MENU']}")
                if doc.get('ISSUE'):
                    text_parts.append(f"Issue: {doc['ISSUE']}")
                if doc.get('EXPECTED'):
                    text_parts.append(f"Solution: {doc['EXPECTED']}")
                if doc.get('NOTE BY DEV'):
                    text_parts.append(f"Dev Note: {doc['NOTE BY DEV']}")
                
                combined_text = " | ".join(text_parts)
                texts.append(combined_text)
                
                # Prepare metadata
                metadata = {
                    'menu': doc.get('MENU', ''),
                    'issue_type': doc.get('ISSUE', ''),
                    'solution': doc.get('EXPECTED', ''),
                    'dev_note': doc.get('NOTE BY DEV', ''),
                    'qa_note': doc.get('NOTE BY QA', ''),
                    'source': 'sipd_csv_data'
                }
                metadatas.append(metadata)
                ids.append(f"doc_{i}")
            
            # Add to vector store (embeddings will be generated automatically)
            self.vector_store.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    async def search_similar(self, query: str, n_results: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar documents using Nebius embeddings"""
        try:
            if not self.vector_store:
                await self.initialize_vector_store()
            
            # Query the vector store
            results = self.vector_store.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            similar_docs = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score
                    similarity = 1 - distance
                    
                    if similarity >= similarity_threshold:
                        similar_docs.append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity': similarity,
                            'rank': i + 1
                        })
            
            logger.info(f"Found {len(similar_docs)} similar documents for query: {query[:50]}...")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    async def get_context_for_query(self, query: str, max_context_length: int = 2000) -> str:
        """Get relevant context for a query using Nebius embeddings"""
        try:
            similar_docs = await self.search_similar(query, n_results=3)
            
            if not similar_docs:
                return "No relevant context found in the knowledge base."
            
            # Build context from similar documents
            context_parts = []
            current_length = 0
            
            for doc in similar_docs:
                metadata = doc['metadata']
                
                # Format context entry
                context_entry = f"""Menu: {metadata.get('menu', 'N/A')}
Issue: {metadata.get('issue_type', 'N/A')}
Solution: {metadata.get('solution', 'N/A')}
Dev Note: {metadata.get('dev_note', 'N/A')}
Similarity: {doc['similarity']:.2f}
---"""
                
                if current_length + len(context_entry) <= max_context_length:
                    context_parts.append(context_entry)
                    current_length += len(context_entry)
                else:
                    break
            
            context = "\n".join(context_parts)
            logger.info(f"Generated context of {len(context)} characters")
            return context
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return "Error retrieving context from knowledge base."
    
    async def update_embedding_cache(self, text: str, embedding: List[float]):
        """Update embedding cache for frequently used queries"""
        text_hash = hash(text)
        self.embedding_cache[text_hash] = {
            'embedding': embedding,
            'text': text,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        # Limit cache size
        if len(self.embedding_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self.embedding_cache.keys(),
                key=lambda k: self.embedding_cache[k]['timestamp']
            )[:100]
            
            for key in oldest_keys:
                del self.embedding_cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        try:
            if not self.vector_store:
                return {'error': 'Vector store not initialized'}
            
            collection_info = self.vector_store.get()
            
            return {
                'collection_name': self.collection_name,
                'total_documents': len(collection_info['ids']) if collection_info['ids'] else 0,
                'embedding_model': self.embedding_client.embedding_model,
                'cache_size': len(self.embedding_cache),
                'vector_store_path': settings.VECTOR_STORE_PATH
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}

# Usage example and integration
async def main():
    """Example usage of Enhanced RAG System"""
    
    # Initialize enhanced RAG system
    rag_system = EnhancedRAGSystem()
    
    # Initialize vector store
    await rag_system.initialize_vector_store()
    
    # Example: Add sample documents
    sample_docs = [
        {
            'MENU': 'Login',
            'ISSUE': 'Tidak bisa login ke sistem SIPD',
            'EXPECTED': 'Periksa username dan password, pastikan koneksi internet stabil',
            'NOTE BY DEV': 'Cek juga apakah browser sudah update',
            'NOTE BY QA': 'Tested on Chrome, Firefox, Edge'
        },
        {
            'MENU': 'DPA',
            'ISSUE': 'Error saat upload DPA',
            'EXPECTED': 'Pastikan file format Excel (.xlsx) dan ukuran maksimal 10MB',
            'NOTE BY DEV': 'Validasi format file di frontend',
            'NOTE BY QA': 'Upload berhasil dengan file valid'
        }
    ]
    
    # Add documents to vector store
    await rag_system.add_documents(sample_docs)
    
    # Search for similar documents
    query = "Saya tidak bisa masuk ke sistem"
    context = await rag_system.get_context_for_query(query)
    
    print(f"Query: {query}")
    print(f"Context: {context}")
    
    # Get system stats
    stats = rag_system.get_stats()
    print(f"RAG System Stats: {stats}")

if __name__ == "__main__":
    asyncio.run(main())