import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from loguru import logger
import numpy as np
from config import settings

class SIPDRAGSystem:
    """Retrieval-Augmented Generation system untuk SIPD Chatbot"""
    
    def __init__(self, collection_name: str = "sipd_knowledge_base"):
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=settings.vector_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "SIPD Help Desk Knowledge Base"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def prepare_documents_from_csv(self, csv_directory: str = "./data/csv") -> List[Dict[str, Any]]:
        """Prepare documents from CSV files for vector storage"""
        documents = []
        
        if not os.path.exists(csv_directory):
            logger.warning(f"CSV directory {csv_directory} not found")
            return documents
            
        for filename in os.listdir(csv_directory):
            if filename.endswith('.csv'):
                try:
                    file_path = os.path.join(csv_directory, filename)
                    df = pd.read_csv(file_path, encoding='utf-8')
                    
                    # Standardize column names
                    df.columns = df.columns.str.strip().str.upper()
                    
                    for idx, row in df.iterrows():
                        issue = str(row.get('ISSUE', '')).strip()
                        expected = str(row.get('EXPECTED', '')).strip()
                        menu = str(row.get('MENU', '')).strip()
                        note_dev = str(row.get('NOTE BY DEV', '') or row.get('NOTE_BY_DEV', '')).strip()
                        note_qa = str(row.get('NOTE BY QA', '') or row.get('NOTE_BY_QA', '')).strip()
                        
                        if issue and issue != 'nan':
                            # Create comprehensive document
                            doc_content = f"Menu: {menu}\nMasalah: {issue}"
                            
                            solution_parts = []
                            if expected and expected != 'nan':
                                solution_parts.append(f"Solusi: {expected}")
                            if note_dev and note_dev != 'nan':
                                solution_parts.append(f"Catatan Developer: {note_dev}")
                            if note_qa and note_qa != 'nan':
                                solution_parts.append(f"Catatan QA: {note_qa}")
                                
                            if solution_parts:
                                doc_content += "\n" + "\n".join(solution_parts)
                            
                            documents.append({
                                'id': f"{filename}_{idx}",
                                'content': doc_content,
                                'metadata': {
                                    'source_file': filename,
                                    'menu': menu,
                                    'issue': issue,
                                    'expected': expected,
                                    'note_dev': note_dev,
                                    'note_qa': note_qa
                                }
                            })
                            
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    
        logger.info(f"Prepared {len(documents)} documents for vector storage")
        return documents
    
    def add_documents_to_vector_store(self, documents: List[Dict[str, Any]]):
        """Add documents to vector store"""
        if not documents:
            logger.warning("No documents to add")
            return
            
        try:
            # Prepare data for ChromaDB
            ids = [doc['id'] for doc in documents]
            contents = [doc['content'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(contents).tolist()
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
    
    def search_similar_documents(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for similar documents based on query"""
        if top_k is None:
            top_k = settings.top_k_results
            
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in vector store
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            similar_docs = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Filter by similarity threshold
                    similarity_score = 1 - distance  # Convert distance to similarity
                    if similarity_score >= settings.similarity_threshold:
                        similar_docs.append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity_score': similarity_score
                        })
            
            logger.info(f"Found {len(similar_docs)} relevant documents for query")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_context_for_query(self, query: str) -> str:
        """Get relevant context for a query"""
        similar_docs = self.search_similar_documents(query)
        
        if not similar_docs:
            return "Maaf, saya tidak menemukan informasi yang relevan dalam database."
            
        # Combine relevant documents into context
        context_parts = []
        for doc in similar_docs[:3]:  # Use top 3 most relevant
            context_parts.append(doc['content'])
            
        context = "\n\n---\n\n".join(context_parts)
        return context
    
    def initialize_knowledge_base(self, csv_directory: str = "./data/csv"):
        """Initialize the knowledge base from CSV files"""
        logger.info("Initializing knowledge base...")
        
        # Check if collection already has data
        try:
            count = self.collection.count()
            if count > 0:
                logger.info(f"Knowledge base already contains {count} documents")
                return
        except:
            pass
            
        # Prepare and add documents
        documents = self.prepare_documents_from_csv(csv_directory)
        if documents:
            self.add_documents_to_vector_store(documents)
            logger.info("Knowledge base initialization completed")
        else:
            logger.warning("No documents found to initialize knowledge base")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'embedding_model': 'all-MiniLM-L6-v2'
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Test the RAG system
    rag = SIPDRAGSystem()
    rag.initialize_knowledge_base()
    
    # Test search
    test_query = "masalah login SIPD"
    context = rag.get_context_for_query(test_query)
    print(f"Context for '{test_query}':")
    print(context)