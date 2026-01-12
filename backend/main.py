import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
from backend.agents import (
    QuantAgent,
    NewsAgent,
    Orchestrator,
    fetch_yfinance_data,
    fetch_news_headlines,
    synthesize_investment_report,
)
import uvicorn

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Explicitly load the .env file located in the backend folder so
# running the app from the project root still picks up the variables.
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

logger.info("Backend initialized with environment variables from backend/.env")

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


class SynthesizeRequest(BaseModel):
    ticker: str
    quant_data: dict
    news_data: list


@app.get("/")
async def root():
    logger.info("Root endpoint hit")
    return {"status": "ok", "engine": "InvestAI Multi-Agent System"}


# ============================================================================
# Tool Endpoints - For testing and observability
# ============================================================================


@app.get("/tools/yfinance/{ticker}")
async def get_yfinance_data(ticker: str):
    """
    Standalone tool endpoint: Fetch yfinance data for a ticker.
    Useful for testing the data collection layer independently.
    """
    logger.info(f"[Endpoint] /tools/yfinance/{ticker} hit")
    try:
        result = fetch_yfinance_data(ticker)
        logger.debug(f"[Endpoint] Successfully fetched yfinance data for {ticker}")
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"[Endpoint] Error fetching yfinance data for {ticker}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch yfinance data: {str(e)}"
        )


@app.get("/tools/news/{ticker}")
async def get_news_data(ticker: str):
    """
    Standalone tool endpoint: Fetch news headlines for a ticker.
    Useful for testing the news collection layer independently.
    """
    logger.info(f"[Endpoint] /tools/news/{ticker} hit")
    try:
        result = fetch_news_headlines(ticker)
        logger.debug(f"[Endpoint] Successfully fetched news headlines for {ticker}")
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"[Endpoint] Error fetching news headlines for {ticker}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch news headlines: {str(e)}"
        )


@app.post("/tools/synthesize")
async def synthesize_report(request: SynthesizeRequest):
    """
    Standalone tool endpoint: Synthesize investment report.
    Accepts quant_data and news_data and generates investment analysis.
    Useful for testing the analysis synthesis layer independently.
    """
    logger.info(
        f"[Endpoint] /tools/synthesize hit for {request.ticker} with {len(request.news_data)} news items"
    )
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        logger.warning("[Endpoint] OPENAI_API_KEY not configured")
        raise HTTPException(
            status_code=400, detail="OPENAI_API_KEY is required for synthesis"
        )

    try:
        result = synthesize_investment_report(
            request.ticker, request.quant_data, request.news_data, openai_key
        )
        logger.debug(f"[Endpoint] Successfully synthesized report for {request.ticker}")
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(
            f"[Endpoint] Error synthesizing report for {request.ticker}: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to synthesize report: {str(e)}"
        )


# ============================================================================
# Main Analysis Endpoint
# ============================================================================


@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    """
    Main orchestration endpoint: Runs the full analysis pipeline.
    1. Quantitative data collection (QuantAgent)
    2. News data collection (NewsAgent)
    3. Report synthesis (Orchestrator)
    """
    ticker = request.ticker.upper()
    logger.info(f"[Endpoint] /analyze hit for ticker: {ticker}")

    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        logger.warning(
            "[Endpoint] OPENAI_API_KEY not configured; returning fallback response"
        )
        # Fallback for UI testing if no key is provided
        return {
            "snapshot": {
                "ticker": ticker,
                "price": 175.45,
                "currency": "USD",
                "pe_ratio": 28.5,
                "debt_to_equity": 0.45,
                "free_cash_flow": 12000000000,
            },
            "analysis": {
                "investment_thesis": "Please provide an OPENAI_API_KEY in the .env file to enable live agentic analysis.",
                "risk_reward": "Risk analysis requires agent synthesis.",
                "portfolio_fit": "Portfolio alignment requires agent synthesis.",
                "bottom_line": "Live intelligence disabled.",
            },
        }

    try:
        logger.debug(f"[Endpoint] Starting analysis pipeline for {ticker}")

        # 1. Quantitative Deep Dive
        logger.debug(f"[Endpoint] Initializing QuantAgent for {ticker}")
        quant = QuantAgent(ticker)
        snapshot_data = quant.get_snapshot()
        logger.debug(f"[Endpoint] QuantAgent snapshot obtained for {ticker}")

        # 2. Market Intelligence
        logger.debug(f"[Endpoint] Initializing NewsAgent for {ticker}")
        news = NewsAgent(ticker)
        headlines = await news.get_recent_headlines()
        logger.debug(f"[Endpoint] NewsAgent headlines obtained for {ticker}")

        # 3. Agentic Orchestration - pass full data to AI
        logger.debug(f"[Endpoint] Initializing Orchestrator for {ticker}")
        orch = Orchestrator(openai_key)
        report = await orch.synthesize_report(
            ticker, snapshot_data["full_data"], headlines  # AI gets all the data
        )
        logger.info(f"[Endpoint] Analysis complete for {ticker}")

        return {
            "snapshot": snapshot_data[
                "display_snapshot"
            ],  # UI gets minimal display data
            "analysis": report,
        }
    except Exception as e:
        logger.error(f"[Endpoint] Error analyzing {ticker}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
