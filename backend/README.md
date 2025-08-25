# AstraFind Backend

Multi-Modal AI Search Agent Backend API

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv myvenv
   myvenv\Scripts\activate  # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment setup:**
   ```bash
   copy .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the server:**
   ```bash
   python app.py
   ```

## API Endpoints

- `GET /` - Health check
- `POST /api/search` - Multi-modal search
- `POST /api/upload` - File upload
- `GET /api/filters` - Available filters
- `GET /api/suggestions` - Search suggestions

## Frontend Integration

The backend is configured to work with your React frontend running on `http://localhost:5173`.

CORS is enabled for local development.
