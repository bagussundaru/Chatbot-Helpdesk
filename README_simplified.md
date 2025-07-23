# Simplified Enhanced SIPD AI Chatbot

## Overview

This is a simplified version of the Enhanced SIPD AI Chatbot that doesn't require the `sentence-transformers` library, which can be difficult to install in some environments. This version maintains most of the core functionality while being easier to set up and run.

## Features

- **Multilingual Support**: Supports Indonesian, English, Javanese, Sundanese, and Malay
- **Intent Classification**: Simple keyword-based classification for common SIPD issues
- **Sentiment Analysis**: Basic sentiment detection to identify user satisfaction
- **Contextual Suggestions**: Provides relevant follow-up suggestions based on user intent
- **Escalation Logic**: Determines when a conversation should be escalated to a human agent
- **Conversation History**: Maintains chat history for each session
- **Modern UI**: Clean, responsive chat interface with language selection

## Simplifications

Compared to the full enhanced chatbot, this version:

1. Doesn't use `sentence-transformers` for embeddings
2. Uses simple keyword-based approaches for language detection, intent classification, and sentiment analysis
3. Uses predefined responses instead of LLM-generated ones
4. Doesn't include the knowledge base functionality
5. Doesn't include the secure API layer

## Requirements

```
fastapi
uvicorn
pydantic
python-dotenv
```

## Running the Application

```bash
python simplified_enhanced_chatbot.py
```

The application will be available at http://localhost:8000

## API Endpoints

- **GET /** - Chat interface
- **POST /chat** - Process chat messages
- **GET /health** - Health check
- **GET /chat/history/{session_id}** - Get conversation history
- **DELETE /chat/history/{session_id}** - Clear conversation history
- **GET /languages** - Get supported languages

## Usage

1. Open http://localhost:8000 in your browser
2. Select your preferred language from the dropdown
3. Type your message in the input field and press Enter or click Send
4. The chatbot will respond with relevant information and suggestions

## Extending the Application

To extend this simplified version:

1. Add more intents and responses in the `sample_responses` dictionary
2. Enhance the keyword lists for intent classification and sentiment analysis
3. Add more languages by updating the system prompts and sample responses
4. Implement more sophisticated logic in the `process_message` method

## Future Improvements

If you want to move towards the full enhanced version:

1. Integrate with a real LLM API when available
2. Add a knowledge base with simpler vector storage
3. Implement more robust language detection
4. Add user authentication and session management
5. Implement logging and monitoring