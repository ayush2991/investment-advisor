import os
import logging
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
from backend.agents import (
    fetch_yfinance_data,
    fetch_news_headlines,
    invalidate_news_cache,
    synthesize_investment_report,
    collect_investment_data,
    analyze_investment,
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
    return {"status": "ok", "engine": "InvestAI Multi-Agent System (OpenAI Agents SDK)"}


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
        logger.info(f"[Endpoint] Fetched yfinance data for {ticker}")
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"[Endpoint] Error fetching yfinance data for {ticker}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch yfinance data: {str(e)}"
        )


@app.get("/tools/news/{ticker}")
async def get_news_data(ticker: str, refresh: bool = False):
    """
    Standalone tool endpoint: Fetch news headlines for a ticker.
    Useful for testing the news collection layer independently.
    """
    logger.info(f"[Endpoint] /tools/news/{ticker} hit")
    try:
        if refresh:
            logger.info(
                f"[Endpoint] refresh=true; invalidating NewsAgent cache for {ticker}"
            )
            removed = invalidate_news_cache(ticker)
            logger.info(f"[Endpoint] invalidate result for {ticker}: removed={removed}")
        result = fetch_news_headlines(ticker)
        logger.info(f"[Endpoint] Fetched news headlines for {ticker}")
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
        result = await synthesize_investment_report(
            request.ticker, request.quant_data, request.news_data, openai_key
        )
        logger.info(f"[Endpoint] Synthesized report for {request.ticker}")
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
    Main orchestration endpoint: Runs the full analysis pipeline using OpenAI Agents SDK.
    1. Data Collection using Data Collector agent
    2. Investment Analysis using Investment Analyst agent

    The agents use the following tools:
    - fetch_yfinance_data: Retrieves comprehensive financial data
    - fetch_news_headlines: Fetches recent market news
    """
    ticker = request.ticker.upper()
    logger.info(f"[Endpoint] /analyze hit for ticker: {ticker}")

    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        logger.warning(
            "[Endpoint] OPENAI_API_KEY not configured; returning fallback response"
        )
        # Fallback for UI testing if no key is provided
        fallback_news = fetch_news_headlines(ticker)
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
            "news": fallback_news,
        }

    try:
        logger.debug(f"[Endpoint] Starting analysis pipeline for {ticker}")

        # Collect financial and news data
        logger.debug(f"[Endpoint] Using agents to collect data for {ticker}")
        financial_data = fetch_yfinance_data(ticker)
        news_data = fetch_news_headlines(ticker)
        logger.debug(f"[Endpoint] Data collection complete for {ticker}")

        # Analyze the collected data using the Investment Analyst agent
        logger.debug(f"[Endpoint] Triggering Investment Analyst agent for {ticker}")
        report = await analyze_investment(
            ticker, financial_data["full_data"], news_data
        )
        logger.info(f"[Endpoint] Analysis complete for {ticker}")

        return {
            "snapshot": financial_data["display_snapshot"],
            "analysis": report,
            "news": news_data,
        }
    except Exception as e:
        logger.error(f"[Endpoint] Error analyzing {ticker}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
