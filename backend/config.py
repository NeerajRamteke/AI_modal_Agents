import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for AstraFind Backend"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:5173,http://localhost:3000').split(',')
    
    # Upload settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx', 'png', 'jpg', 'jpeg'}
    
    # API settings
    API_VERSION = 'v1'
    API_PREFIX = '/api'
    
    # Search settings
    MAX_RESULTS = 50
    DEFAULT_RESULTS = 10
    SEARCH_TIMEOUT = 30  # seconds
    
    # External API keys (add your actual keys here)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    SERP_API_KEY = os.environ.get('SERP_API_KEY')
