# SIPD Helpdesk Chatbot - Vercel Deployment

## Overview
This is a simplified version of the SIPD Helpdesk Chatbot optimized for deployment on Vercel using serverless functions.

## Features
- 🤖 Intelligent chatbot for SIPD helpdesk support
- 🌐 Multi-language support (Indonesian & English)
- 📱 Responsive web interface
- ⚡ Serverless deployment on Vercel
- 🎨 Modern UI with SIPD branding

## Project Structure
```
├── api/
│   └── index.py          # Main FastAPI application for Vercel
├── logo/
│   └── SIPD.svg          # SIPD logo
├── vercel.json           # Vercel configuration
├── requirements_vercel.txt # Simplified dependencies
└── README_VERCEL.md      # This file
```

## Deployment to Vercel

### Prerequisites
- Vercel account
- Git repository

### Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Deploy to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will automatically detect the configuration

3. **Configuration**
   - Build Command: Leave empty (auto-detected)
   - Output Directory: Leave empty
   - Install Command: `pip install -r requirements_vercel.txt`

### Environment Variables
No environment variables are required for the basic version.

## Local Development

1. **Install dependencies**
   ```bash
   pip install -r requirements_vercel.txt
   ```

2. **Run locally**
   ```bash
   uvicorn api.index:app --reload --port 8000
   ```

3. **Access the application**
   Open http://localhost:8000 in your browser

## API Endpoints

- `GET /` - Chat interface
- `POST /chat` - Send message to chatbot
- `GET /health` - Health check
- `GET /history` - Get conversation history
- `DELETE /history` - Clear conversation history
- `GET /languages` - Get supported languages
- `GET /logo/SIPD.svg` - SIPD logo

## Features

### Chatbot Capabilities
- **Intent Classification**: Automatically categorizes user queries
- **Language Detection**: Supports Indonesian and English
- **Smart Responses**: Context-aware responses for common SIPD issues
- **Escalation**: Automatic escalation for urgent issues
- **Suggestions**: Quick reply suggestions for common queries

### Supported Issue Categories
- Login problems
- Access issues
- Data-related queries
- System errors
- General support

## Customization

### Adding New Responses
Edit the `sample_responses` dictionary in `api/index.py`:

```python
self.sample_responses = {
    "id": {
        "new_category": "Response in Indonesian",
        # ... other categories
    },
    "en": {
        "new_category": "Response in English",
        # ... other categories
    }
}
```

### Modifying UI
The HTML/CSS is embedded in the `get_chat_interface()` function in `api/index.py`.

### Adding Languages
1. Add language code to `detect_language()` method
2. Add responses in `sample_responses`
3. Update `get_supported_languages()` endpoint

## Performance Considerations

- **Cold Starts**: First request may be slower due to serverless cold starts
- **Memory Usage**: Optimized for Vercel's memory limits
- **Response Time**: Typically under 1 second for most queries

## Limitations

- No persistent database (conversation history is in-memory)
- No advanced AI/ML features (simplified for serverless)
- Limited to Vercel's execution time limits

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check `requirements_vercel.txt` for correct package versions
   - Ensure Python version compatibility

2. **Runtime Errors**
   - Check Vercel function logs
   - Verify all imports are available

3. **Logo Not Loading**
   - Ensure `logo/SIPD.svg` exists in repository
   - Check file path in `get_logo()` function

### Getting Help

- Check Vercel documentation: https://vercel.com/docs
- Review FastAPI documentation: https://fastapi.tiangolo.com
- Check application logs in Vercel dashboard

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

---

**Note**: This is a simplified version optimized for Vercel deployment. For advanced features like AI integration, database persistence, or complex workflows, consider using a different hosting platform or upgrading to Vercel Pro.