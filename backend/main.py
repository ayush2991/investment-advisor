import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import QuantAgent, NewsAgent, Orchestrator
import uvicorn

load_dotenv()

app = FastAPI(title="InvestAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    ticker: str

@app.get("/")
async def root():
    return {"status": "ok", "engine": "InvestAI Multi-Agent System"}

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    ticker = request.ticker.upper()
    openai_key = os.getenv("OPEN_AI_KEY")
    
    if not openai_key:
        # Fallback for UI testing if no key is provided
        return {
            "snapshot": {
                "ticker": ticker,
                "price": 175.45,
                "currency": "USD",
                "pe_ratio": 28.5,
                "debt_to_equity": 0.45,
                "free_cash_flow": 12000000000
            },
            "analysis": {
                "investment_thesis": "Please provide an OPEN_AI_KEY in the .env file to enable live agentic analysis.",
                "risk_reward": "Risk analysis requires agent synthesis.",
                "portfolio_fit": "Portfolio alignment requires agent synthesis.",
                "bottom_line": "Live intelligence disabled."
            }
        }

    try:
        # 1. Quantitative Deep Dive
        quant = QuantAgent(ticker)
        snapshot = quant.get_snapshot()
        
        # 2. Market Intelligence
        news = NewsAgent(ticker)
        headlines = await news.get_recent_headlines()
        
        # 3. Agentic Orchestration
        orch = Orchestrator(openai_key)
        report = await orch.synthesize_report(ticker, snapshot, headlines)
        
        return {
            "snapshot": snapshot,
            "analysis": report
        }
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
