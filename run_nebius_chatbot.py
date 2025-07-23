#!/usr/bin/env python3
# Script untuk menjalankan SIPD Nebius Chatbot
# Menyediakan interface yang mudah untuk memulai chatbot dengan berbagai konfigurasi

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import uvicorn
    from nebius_chatbot_config import config, validate_config
    from nebius_chatbot import app, chatbot
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Pastikan semua dependencies sudah terinstall dengan menjalankan:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment() -> bool:
    """Check if environment is properly configured"""
    print("🔍 Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {sys.version.split()[0]}")
    
    # Check required environment variables
    required_vars = {
        'NEBIUS_API_KEY': 'Nebius AI API key',
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")
        else:
            print(f"✅ {var} configured")
    
    if missing_vars:
        print("\n❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPastikan file .env sudah dikonfigurasi dengan benar.")
        return False
    
    # Validate configuration
    if not validate_config():
        print("❌ Configuration validation failed")
        return False
    
    print("✅ Configuration valid")
    return True

def setup_directories():
    """Setup required directories"""
    print("📁 Setting up directories...")
    
    directories = [
        config.vector_store_path,
        "logs",
        "data/processed"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")

async def test_nebius_connection() -> bool:
    """Test connection to Nebius AI"""
    print("🔗 Testing Nebius AI connection...")
    
    try:
        # Initialize chatbot to test connection
        await chatbot.initialize()
        
        # Test basic functionality
        health_status = await chatbot.get_health_status()
        
        if health_status.get('nebius_connection'):
            print("✅ Nebius AI connection successful")
            return True
        else:
            print("❌ Nebius AI connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Nebius AI connection error: {e}")
        return False

def print_startup_info():
    """Print startup information"""
    print("\n" + "="*60)
    print("🤖 SIPD NEBIUS CHATBOT")
    print("="*60)
    print(f"Model: {config.nebius_model_id}")
    print(f"Embedding Model: {config.nebius_embedding_model}")
    print(f"RAG Enabled: {config.rag_enabled}")
    print(f"Caching Enabled: {config.enable_caching}")
    print(f"Rate Limiting: {config.enable_rate_limiting}")
    print(f"Max Tokens: {config.max_tokens}")
    print(f"Temperature: {config.temperature}")
    print("="*60)

def print_access_info(host: str, port: int):
    """Print access information"""
    print("\n🚀 Chatbot is running!")
    print(f"\n📱 Web Interface: http://{host}:{port}")
    print(f"🔗 API Docs: http://{host}:{port}/docs")
    print(f"❤️ Health Check: http://{host}:{port}/health")
    print(f"📊 Stats: http://{host}:{port}/stats")
    
    if config.enable_metrics:
        print(f"📈 Metrics: http://{host}:{config.metrics_port}/metrics")
    
    print("\n💡 Tips:")
    print("   - Gunakan Ctrl+C untuk menghentikan server")
    print("   - Buka web interface untuk mulai chat")
    print("   - Check /health endpoint untuk status sistem")
    print("   - Check /stats untuk statistik penggunaan")
    print("\n" + "="*60)

async def run_chatbot(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    test_mode: bool = False
):
    """Run the chatbot server"""
    
    if test_mode:
        print("🧪 Running in test mode...")
        
        # Test connection only
        connection_ok = await test_nebius_connection()
        if connection_ok:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Tests failed!")
            return False
    
    # Normal run mode
    print_startup_info()
    
    # Test connection before starting server
    connection_ok = await test_nebius_connection()
    if not connection_ok:
        print("❌ Cannot start server - Nebius connection failed")
        return False
    
    print_access_info(host, port)
    
    # Start the server
    config_dict = {
        "app": "nebius_chatbot:app",
        "host": host,
        "port": port,
        "reload": reload,
        "log_level": config.log_level.lower(),
        "access_log": True
    }
    
    try:
        await uvicorn.run(**config_dict)
    except KeyboardInterrupt:
        print("\n👋 Shutting down chatbot...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        return False
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="SIPD Nebius Chatbot Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_nebius_chatbot.py                    # Run with default settings
  python run_nebius_chatbot.py --port 8080        # Run on port 8080
  python run_nebius_chatbot.py --reload           # Run with auto-reload
  python run_nebius_chatbot.py --test             # Test connection only
  python run_nebius_chatbot.py --host 127.0.0.1   # Run on localhost only
        """
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test connection and configuration only"
    )
    
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip environment checks (not recommended)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 Starting SIPD Nebius Chatbot...")
    
    # Environment checks
    if not args.skip_checks:
        if not check_environment():
            print("\n❌ Environment check failed. Use --skip-checks to bypass (not recommended).")
            sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Run the chatbot
    try:
        success = asyncio.run(run_chatbot(
            host=args.host,
            port=args.port,
            reload=args.reload,
            test_mode=args.test
        ))
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()