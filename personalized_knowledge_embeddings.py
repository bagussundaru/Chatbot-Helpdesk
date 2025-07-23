# Personalized Knowledge Embeddings for SIPD AI Chatbot
# Implementasi sistem RAG dengan personalisasi untuk domain SIPD

import os
import json
import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from loguru import logger
from datetime import datetime
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from config import settings

class PersonalizedKnowledgeEmbeddings:
    """Personalized Knowledge Embeddings untuk domain-specific data"""
    
    def __init__(self, collection_name: str = "sipd_knowledge_base"):
        self.collection_name = collection_name
        self.vector_store = None
        self.embedding_model = None
        self.embedding_cache = {}
        self.max_cache_size = 1000
        self.similarity_threshold = 0.6  # Default similarity threshold
        
        # Setup logging
        logger.add("logs/personalized_knowledge.log", rotation="10 MB", level="INFO")
    
    async def initialize(self) -> bool:
        """Initialize embedding system"""
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize ChromaDB client
            chroma_client = chromadb.PersistentClient(
                path=settings.vector_db_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection
            self.vector_store = chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "SIPD Knowledge Base with Personalized Embeddings"}
            )
            
            logger.info(f"Personalized Knowledge Embeddings initialized with collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            return False
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            # Check cache first
            if text in self.embedding_cache:
                return self.embedding_cache[text]
            
            # Generate embedding
            embedding = self.embedding_model.encode(text).tolist()
            
            # Update cache
            self._update_cache(text, embedding)
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
    
    def _update_cache(self, text: str, embedding: List[float]):
        """Update embedding cache with size limit"""
        # Add to cache
        self.embedding_cache[text] = embedding
        
        # Check cache size
        if len(self.embedding_cache) > self.max_cache_size:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to knowledge base"""
        try:
            if not self.vector_store or not self.embedding_model:
                await self.initialize()
            
            ids = []
            texts = []
            metadatas = []
            embeddings = []
            
            for doc in documents:
                # Generate unique ID
                doc_id = doc.get("id", str(hash(doc.get("content", "") + str(datetime.now()))))
                
                # Prepare content
                content = self._prepare_document_content(doc)
                
                # Generate embedding
                embedding = self._generate_embedding(content)
                
                # Add to lists
                ids.append(doc_id)
                texts.append(content)
                metadatas.append({
                    "menu": doc.get("menu", ""),
                    "issue": doc.get("issue", ""),
                    "expected": doc.get("expected", ""),
                    "source": doc.get("source", "knowledge_base"),
                    "timestamp": datetime.now().isoformat()
                })
                embeddings.append(embedding)
            
            # Add to vector store
            self.vector_store.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"Added {len(documents)} documents to knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def _prepare_document_content(self, doc: Dict[str, Any]) -> str:
        """Prepare document content for embedding"""
        # Combine fields with appropriate formatting
        content_parts = []
        
        if "menu" in doc and doc["menu"]:
            content_parts.append(f"Menu: {doc['menu']}")
        
        if "issue" in doc and doc["issue"]:
            content_parts.append(f"Issue: {doc['issue']}")
        
        if "expected" in doc and doc["expected"]:
            content_parts.append(f"Expected: {doc['expected']}")
        
        if "note_by_dev" in doc and doc["note_by_dev"]:
            content_parts.append(f"Dev Note: {doc['note_by_dev']}")
        
        if "note_by_qa" in doc and doc["note_by_qa"]:
            content_parts.append(f"QA Note: {doc['note_by_qa']}")
        
        if "content" in doc and doc["content"]:
            content_parts.append(doc["content"])
        
        # Join all parts
        return "\n\n".join(content_parts)
    
    async def search_similar(self, query: str, n_results: int = 5, threshold: float = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if not self.vector_store or not self.embedding_model:
                await self.initialize()
            
            # Use provided threshold or default
            similarity_threshold = threshold if threshold is not None else self.similarity_threshold
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Query the vector store
            results = self.vector_store.query(
                query_embeddings=[query_embedding],
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
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    # Normalize to 0-1 range where 1 is most similar
                    similarity = 1 - min(distance, 2) / 2
                    
                    # Apply threshold
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
    
    async def get_context(self, query: str, n_results: int = 5, threshold: float = None) -> str:
        """Get context for query from knowledge base"""
        try:
            # Search for similar documents
            similar_docs = await self.search_similar(query, n_results, threshold)
            
            if not similar_docs:
                return ""
            
            # Format context
            context_parts = []
            
            for i, doc in enumerate(similar_docs):
                # Format metadata
                metadata = doc.get('metadata', {})
                menu = metadata.get('menu', '')
                issue = metadata.get('issue', '')
                expected = metadata.get('expected', '')
                
                # Format context entry
                context_entry = f"### Dokumen {i+1} (Similarity: {doc['similarity']:.2f})\n"
                
                if menu:
                    context_entry += f"Menu: {menu}\n"
                
                if issue:
                    context_entry += f"Issue: {issue}\n"
                
                if expected:
                    context_entry += f"Solution: {expected}\n"
                
                # Add content
                context_entry += f"\n{doc['content']}\n"
                
                context_parts.append(context_entry)
            
            # Join all parts
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return ""
    
    async def initialize_from_csv(self, csv_file_path: str, id_column: str = None) -> bool:
        """Initialize knowledge base from CSV file"""
        try:
            # Check if file exists
            if not os.path.exists(csv_file_path):
                logger.error(f"CSV file not found: {csv_file_path}")
                return False
            
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            
            # Standardize column names
            df.columns = [col.upper() for col in df.columns]
            
            # Map standard columns
            column_mapping = {
                'MENU': 'menu',
                'ISSUE': 'issue',
                'EXPECTED': 'expected',
                'NOTE BY DEV': 'note_by_dev',
                'NOTE BY QA': 'note_by_qa'
            }
            
            # Prepare documents
            documents = []
            
            for _, row in df.iterrows():
                doc = {}
                
                # Add ID if specified
                if id_column and id_column in df.columns:
                    doc['id'] = str(row[id_column])
                
                # Map columns
                for csv_col, doc_col in column_mapping.items():
                    if csv_col in df.columns and not pd.isna(row[csv_col]):
                        doc[doc_col] = str(row[csv_col])
                
                # Add to documents
                documents.append(doc)
            
            # Add documents to knowledge base
            success = await self.add_documents(documents)
            
            if success:
                logger.info(f"Initialized knowledge base from CSV: {csv_file_path} with {len(documents)} documents")
            
            return success
            
        except Exception as e:
            logger.error(f"Error initializing from CSV: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            if not self.vector_store:
                await self.initialize()
            
            # Get collection count
            count = self.vector_store.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model.__class__.__name__ if self.embedding_model else None,
                "embedding_dimension": 384,  # Default for all-MiniLM-L6-v2
                "cache_size": len(self.embedding_cache),
                "vector_store_path": settings.vector_db_path
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "error": str(e)
            }

# Example usage
if __name__ == "__main__":
    async def test_embeddings():
        # Initialize embeddings
        embeddings = PersonalizedKnowledgeEmbeddings()
        await embeddings.initialize()
        
        # Add sample documents
        sample_docs = [
            {
                "menu": "Login",
                "issue": "Tidak bisa login ke SIPD",
                "expected": "Pastikan username dan password benar, clear cache browser, dan coba browser lain.",
                "note_by_dev": "Masalah umum terkait cache browser atau cookies."
            },
            {
                "menu": "DPA",
                "issue": "Error saat menyimpan DPA",
                "expected": "Pastikan semua field mandatory terisi dan format data sesuai.",
                "note_by_qa": "Validasi field sering menjadi masalah."
            },
            {
                "menu": "Laporan",
                "issue": "Tidak bisa export laporan ke Excel",
                "expected": "Coba export dengan data lebih sedikit atau per periode.",
                "note_by_dev": "Masalah timeout saat data terlalu besar."
            }
        ]
        
        await embeddings.add_documents(sample_docs)
        
        # Search for similar documents
        query = "Saya tidak bisa login ke SIPD"
        similar_docs = await embeddings.search_similar(query)
        
        print("\nSimilar documents:")
        for doc in similar_docs:
            print(f"Similarity: {doc['similarity']:.2f}")
            print(f"Content: {doc['content'][:100]}...")
            print(f"Metadata: {doc['metadata']}")
            print()
        
        # Get context
        context = await embeddings.get_context(query)
        print("\nContext:")
        print(context)
        
        # Get stats
        stats = await embeddings.get_stats()
        print("\nStats:")
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    # Run the test
    asyncio.run(test_embeddings())