# Investment Advisor

An AI-powered investment analysis platform that combines quantitative data, market intelligence, and AI synthesis to provide comprehensive investment reports. The current UI ships with a midnight/ember palette, glassy cards, and Space Grotesk typography for a modern trading-desk feel.

## Features

- **Quantitative Analysis**: Real-time financial data from yfinance with 4-hour caching
- **Market Intelligence**: News aggregation for tickers (mock implementation)
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
- OpenAI API key (set in `backend/.env`)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r ../requirements.txt
```

### Backend Run

```bash
source venv/bin/activate
uvicorn main:app --reload --port 8000
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

Frontend will be available at `http://localhost:5173/` (Vite will auto-bump to 5174/5175 if ports are occupied).

### Local Dev Loop

- Start backend: `./venv/bin/python -m uvicorn backend.main:app --reload --port 8000`
- Start frontend: `npm run dev` and open the printed localhost port
- If the Analysis view shows "Failed to fetch", confirm the backend is running on `:8000`

## Environment Variables

Create `backend/.env`:

```
OPENAI_API_KEY=sk-...
```

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
    "ticker": "AAPL",
    "price": 259.37,
    "currency": "USD",
    "pe_ratio": 34.72,
    "debt_to_equity": 152.41,
    "market_cap": 3832542658560,
    "analyst_ratings": {
      "recommendation": "buy",
      "target_mean": 287.83,
      "number_of_analysts": 41
    }
  },
  "analysis": {
    "investment_thesis": "...",
    "risk_reward": "...",
    "portfolio_fit": "...",
    "bottom_line": "..."
  }
}
```

## UI Notes

- Dark, glassmorphic surface with ember/orange accents and white text
- Space Grotesk for titles and body
- Sidebar/header updated with glow accents; cards use gradient fills
- Dashboard: glass stat cards, accent chart bars, refined lists
- Analysis: accented search/CTA, glass report/snapshot cards

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

- **Backend**: fastapi, uvicorn, yfinance, openai, diskcache, python-dotenv
- **Frontend**: react, lucide-react, vite

## Development Notes

- Backend uses uvicorn with `--reload` for hot-reloading during development
- Diskcache automatically handles TTL expiration; no manual cleanup needed
- Frontend state persists via localStorage; clear to reset Analysis view
- Mock news implementation; replace `NewsAgent.get_recent_headlines()` with Tavily/Serper API

## Future Enhancements

- Replace mock news with real API (Tavily, Serper, etc.)
- Add cache statistics endpoint (`GET /cache/stats`)
- Implement Redis for multi-instance deployments
- Add portfolio recommendations based on user risk profile
- Extend analysis with technical indicators and sentiment analysis
