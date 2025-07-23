import os
import sys
from loguru import logger
import json
from bitext_processor import BitextDataProcessor
from nebius_embedding_integration import EnhancedRAGSystem

def setup_logging():
    """Setup logging configuration"""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/bitext_rag_test.log", rotation="10 MB", level="DEBUG")

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        "data/bitext",
        "data/processed",
        "logs"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def load_processed_data(file_path):
    """Load processed training data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} training examples from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return []

def convert_to_rag_documents(training_data):
    """Convert training data to RAG document format"""
    documents = []
    
    for item in training_data:
        messages = item.get('messages', [])
        if len(messages) >= 2:
            user_message = next((m['content'] for m in messages if m['role'] == 'user'), '')
            assistant_message = next((m['content'] for m in messages if m['role'] == 'assistant'), '')
            
            if user_message and assistant_message:
                # Extract menu if available (format: "Saya mengalami masalah di menu {menu}: {issue}")
                menu = ""
                issue = user_message
                if "Saya mengalami masalah di menu" in user_message:
                    parts = user_message.split(":", 1)
                    if len(parts) > 1:
                        menu_parts = parts[0].split("menu", 1)
                        if len(menu_parts) > 1:
                            menu = menu_parts[1].strip()
                        issue = parts[1].strip()
                
                # Create document
                document = {
                    "MENU": menu,
                    "ISSUE": issue,
                    "EXPECTED": assistant_message,
                    "NOTE_BY_DEV": "",
                    "NOTE_BY_QA": ""
                }
                documents.append(document)
    
    logger.info(f"Converted {len(documents)} training examples to RAG documents")
    return documents

async def test_rag_with_bitext_data():
    """Test RAG system with Bitext data"""
    # Initialize RAG system
    rag_system = EnhancedRAGSystem()
    
    # Load processed Bitext data
    bitext_data_path = "data/processed/bitext_training_data.json"
    bitext_data = load_processed_data(bitext_data_path)
    
    if not bitext_data:
        logger.error("No Bitext data found. Please run integrate_bitext_dataset.py first.")
        return
    
    # Convert to RAG documents
    rag_documents = convert_to_rag_documents(bitext_data)
    
    # Add documents to RAG system
    logger.info("Adding documents to RAG system...")
    await rag_system.add_documents(rag_documents)
    
    # Test queries
    test_queries = [
        "How do I reset my password?",
        "I need to cancel my order",
        "Where is my refund?",
        "I can't log into my account",
        "How do I update my shipping address?"
    ]
    
    logger.info("Testing RAG system with sample queries...")
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        context = await rag_system.get_context_for_query(query)
        logger.info(f"Context: {context[:200]}..." if len(context) > 200 else f"Context: {context}")

def main():
    """Main function to test Bitext RAG integration"""
    import asyncio
    
    setup_logging()
    logger.info("Starting Bitext RAG integration test")
    
    ensure_directories()
    
    # Run async function in sync context
    asyncio.run(test_rag_with_bitext_data())
    
    logger.info("Test completed")

if __name__ == "__main__":
    main()