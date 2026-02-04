# PortfolioPulse Investment Advisor

PortfolioPulse is a high-performance, single-stock and portfolio analysis platform designed for modern investors. It leverages a sophisticated multi-agent AI pipeline to synthesize real-time market data, technical signals, and curated news into actionable institutional-grade insights.

## 🚀 Overview

- PortfolioPulse bridges the gap between raw financial data and human-readable analysis. By orchestrating specialized AI agents, it provides:
- **Deep Fundamentals**: Automated valuation and growth analysis using `yfinance`.
- **Technical Momentum**: Quantitative trend and volatility tracking.
- **Agentic Market Chatter**: Intelligent search query generation and news selection using Tavily.
- **Strategic Portfolio Comparison**: Detailed head-to-head analysis of two different investment strategies with AI-driven scenario modeling.
- **Portfolio Intelligence**: Advanced risk-return metrics (Sharpe, HHI, Beta) and AI-driven rebalancing suggestions.
- **Premium Experience**: A sleek, glassmorphic UI built for speed and clarity.

---

## 🛠 Tech Stack

### Backend
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.9+)
- **Data Source**: `yfinance` for real-time and historical market data.
- **AI Orchestration**: `openai-agents` SDK for managing specialized LLM personas.
- **Caching**: `cachetools` (TTLCache) for optimized API performance and cost management.
- **Search**: [Tavily API](https://tavily.com/) for high-signal financial news retrieval.

### Frontend
- **UI/UX**: Vanilla JavaScript, Semantic HTML5, and Modern CSS.
- **Design System**: Glassmorphism with responsive layouts and real-time status updates.
- **Visuals**: Dynamic progress indicators for multi-stage AI analysis.

---

## 📂 Project Structure

```text
investment-advisor/
├── app/
│   ├── main.py          # Core FastAPI application & Agent logic
│   ├── static/          # Frontend assets
│   │   ├── app.js       # UI logic and API interaction
│   │   └── styles.css   # Modern glassmorphic design system
│   └── templates/
│       └── index.html   # Main application shell
├── requirements.txt     # Python dependencies
└── README.md            # You are here
```

---

## 🧠 Agentic Architecture

The "brain" of PortfolioPulse is a pipeline of 7 specialized agents:

1.  **News Query Agent**: Analyzes the stock context to generate 3-5 high-signal search queries.
2.  **News Selection Agent**: Filters raw search results from Tavily to select the most impactful articles.
3.  **Fundamentals Agent**: Deep dives into Valuation, Growth, and Profitability metrics.
4.  **Technicals Agent**: Analyzes price action, moving averages, and volatility.
5.  **News Agent**: Synthesizes selected articles into a concise sentiment-aware briefing.
6.  **Synthesis Agent**: The "Lead Advisor" that merges all previous outputs into a final investor briefing.
7.  **Portfolio Analyst**: Evaluates diversification, factor exposures, and suggests optimized allocations.

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.9 or higher
- [Tavily API Key](https://tavily.com/) (for Market Chatter)
- [OpenAI API Key](https://platform.openai.com/) (for Research Synthesis)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd investment-advisor
    ```

2.  **Set up Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    # OR: .venv\Scripts\activate  # Windows
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

Create a `.env` file in the root directory or export variables directly:

```bash
export OPENAI_API_KEY="your_openai_key"
export TAVILY_API_KEY="your_tavily_key"
export OPENAI_MODEL="gpt-4o"  # Optional: Default is gpt-5-nano (if available)
export LOG_LEVEL="INFO"        # Optional: DEBUG for verbose agent logs
```

### Running the App

```bash
uvicorn app.main:app --reload
```
Navigate to `http://127.0.0.1:8000` in your browser.

---

## 📈 Development Guide

### Adding a New Agent
To add a new capability:
1.  Define a new `BaseModel` in `app/main.py` for the agent's output.
2.  Create a `_build_<name>_agent()` function.
3.  Integrate the agent into the `_openai_stock_brief` or `_openai_portfolio_brief` workflow.

### Caching Strategy
The app uses tiered `TTLCache` to minimize latency and API costs:
- **Market Info**: 30 minutes
- **Price History**: 30 minutes
- **News Results**: 15 minutes
- **AI Synthesis**: 30 minutes

### Deployment
PortfolioPulse is ready to be deployed on platforms like **Render**, **Railway**, or **Heroku**. Ensure the environment variables are configured in your dashboard.

---

## ⚖️ Disclaimer
*PortfolioPulse is an educational tool and does not constitute financial advice. Always perform your own due diligence or consult with a certified financial advisor before making investment decisions.*
