# VIP Merchant Command Center

This package separates the Decision Intelligence Platform into a live-updating, premium Web App (Frontend) and a fast HTTP/WebSocket Python API (Backend).

## 1. Quick Start

### Python Backend (FastAPI)
The backend loads the pre-trained `joblib` models and opens a `POST /predict_live` alongside a WebSocket `/ws/live_visitors` serving dummy live visitors.

1. Open a new terminal.
2. `cd backend`
3. If you haven't yet, install requirements: `pip install -r requirements.txt`
4. Run the server: `uvicorn main:app --reload --port 8000`

### React Dashboard (Vite)
A gorgeous, modern dark-mode application pulling AI stats via WebSocket.

1. Open a second terminal.
2. `cd frontend`
3. Run the development server: `npm run dev`
4. Open the localhost URL it provides in your browser (usually `http://localhost:5173`).

## 2. Storefront Tracking Script
Add `<script src="storefront_tracking.js"></script>` to any e-commerce storefront. It quietly monitors clicks and pages via Session Storage and pushes updates to the engine every 12 seconds exactly as requested.
