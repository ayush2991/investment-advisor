# Investment Advisor

An AI-powered investment analysis platform that combines quantitative data, market intelligence, and AI synthesis to provide comprehensive investment reports. The current UI ships with a midnight/ember palette, glassy cards, and Space Grotesk typography for a modern trading-desk feel.

## Features

- **Quantitative Analysis**: Real-time financial data from yfinance with 4-hour caching
- **Market Intelligence**: Real news via Tavily when `TAVILY_API_KEY` is configured; graceful fallback to mock headlines
- **AI Synthesis**: OpenAI-powered investment thesis generation
- **Smart Caching**: Multi-layer caching with diskcache (yfinance, news, orchestration) — 4-hour TTL
- **State Persistence**: Analysis results persist across tab navigation (localStorage)
- **Fast Responses**: Cached analysis responses return in <50ms vs 50+ seconds for fresh API calls

## Architecture

### Backend (FastAPI)

- **`backend/main.py`**: FastAPI server with `/analyze` endpoint
- **`backend/agents.py`**: Three agent classes:
  - `QuantAgent`: Fetches yfinance data (cached 4 hours)
  - `NewsAgent`: Aggregates news headlines (cached 4 hours)
  - `Orchestrator`: Synthesizes OpenAI analysis (cached 4 hours)
- **Cache**: Diskcache (SQLite-backed) for all agent results

### Frontend (React + Vite)

- **`src/views/Analysis.jsx`**: Main analysis view with localStorage persistence
- **State Management**: Saves ticker, analysis, and error state to localStorage
- **Navigation**: Analysis view persists when navigating to other tabs and back

## Quickstart

### Prerequisites

- Python 3.12+
- Node.js 18+
- OpenAI API key (required)
- Tavily API key (optional, for real news)

### Backend Setup

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

Create `backend/.env`:

```
OPENAI_API_KEY=sk-...
# Optional: enable real news via Tavily
TAVILY_API_KEY=tvly-...
```

### Backend Run

```bash
source venv/bin/activate
python -m uvicorn backend.main:app --reload --port 8000
```

Backend will be available at `http://127.0.0.1:8000/`

### Frontend Setup

```bash
npm install
```

### Frontend Run

```bash
npm run dev
```

Frontend will be available at `http://localhost:5173/`

### Local Dev Loop

1. Start backend: `source venv/bin/activate && python -m uvicorn backend.main:app --reload --port 8000`
2. Start frontend: `npm run dev` and open the printed localhost port
3. If the Analysis view shows "Failed to fetch", confirm the backend is running on port 8000

## API Endpoints

### `GET /`

Health check

```bash
curl http://127.0.0.1:8000/
```

### `POST /analyze`

Analyze a ticker

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL"}' \
  http://127.0.0.1:8000/analyze
```

**Response:**

```json
{
  "snapshot": {
    /* minimal display snapshot */
  },
  "analysis": {
    /* synthesized report */
  },
  "news": [
    {
      "title": "Apple Q4 beats expectations",
      "source": "finance.yahoo.com",
      "url": "https://...",
      "sentiment": "neutral"
    }
  ]
}
```

## UI Notes

- Dark, glassmorphic surface with ember/orange accents and white text
- Space Grotesk for titles and body
- Sidebar/header updated with glow accents; cards use gradient fills
- Dashboard: glass stat cards, accent chart bars, refined lists
- Analysis: accented search/CTA, glass report/snapshot cards
- Market News card shows headlines via Tavily when configured

## Caching

### Three-Layer Caching Strategy

1. **Diskcache (4-hour TTL)**

   - Backend storage: `backend/agent_cache/cache.db` (SQLite)
   - Covers: yfinance data, news headlines, OpenAI synthesis
   - Automatic expiration after 4 hours
   - Survives process restarts and code reloads

2. **Frontend State (localStorage)**
   - Storage key: `analysisState`
   - Persists: ticker, analysis results, errors
   - Survives page refreshes and tab navigation

### Cache Behavior

- **First request for ticker**: ~50 seconds (live APIs)
- **Subsequent requests within 4 hours**: <50ms (diskcache hit)
- **After 4 hours**: Fresh API calls made; cache refreshed

### Clear Cache

**Backend (diskcache):**

```bash
rm -rf backend/agent_cache/
```

**Frontend (localStorage):**

```javascript
localStorage.removeItem("analysisState");
```

## File Structure

```
investment-advisor/
├── backend/
│   ├── agents.py          # QuantAgent, NewsAgent, Orchestrator
│   ├── main.py            # FastAPI server
│   ├── agent_cache/       # Diskcache directory (auto-created)
│   └── .env               # OpenAI API key
├── src/
│   ├── views/
│   │   └── Analysis.jsx   # Main analysis UI with state persistence
│   ├── components/        # Reusable React components
│   ├── App.jsx
│   └── index.css
├── requirements.txt       # Python dependencies
├── package.json           # Node.js dependencies
└── vite.config.js         # Vite configuration
```

## Key Dependencies

- **Backend**: fastapi, uvicorn, yfinance, openai, openai-agents, tavily-python, diskcache, python-dotenv, pydantic, pandas
- **Frontend**: react (v19), react-dom, lucide-react, vite (v7)

## Development Notes

- Backend uses uvicorn with `--reload` for hot-reloading during development
- Diskcache automatically handles TTL expiration; no manual cleanup needed
- Frontend state persists via localStorage; clear to reset Analysis view
- Real news supported via Tavily when `TAVILY_API_KEY` is set; otherwise mock headlines are returned

## Future Enhancements

- Replace mock news with real API (Tavily, Serper, etc.)
- Add cache statistics endpoint (`GET /cache/stats`)
- Implement Redis for multi-instance deployments
- Add portfolio recommendations based on user risk profile
- Extend analysis with technical indicators and sentiment analysis
