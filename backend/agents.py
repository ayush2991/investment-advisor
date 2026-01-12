import os
import yfinance as yf
import openai
from typing import List, Dict, Optional
import json
import logging
import time
from pathlib import Path
import diskcache

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TTL for all caching (4 hours in seconds)
CACHE_TTL = 14400

# Initialize diskcache for agent tool calls
_agent_cache_dir = Path(__file__).parent / "agent_cache"
_agent_cache = diskcache.Cache(str(_agent_cache_dir))


def _get_cache_key(agent_name: str, ticker: str) -> str:
    """Generate a cache key for agent results."""
    return f"{agent_name}:{ticker}"


def _get_cached_result(key: str) -> Optional[Dict]:
    """Retrieve a cached result from diskcache if it exists and is not expired."""
    try:
        result = _agent_cache.get(key)
        if result is not None:
            logger.debug(f"Cache HIT for key: {key}")
            return result
        else:
            logger.debug(f"Cache MISS for key: {key}")
            return None
    except Exception as e:
        logger.warning(f"Cache retrieval error for key {key}: {e}")
        return None


def _set_cached_result(key: str, result: Dict) -> None:
    """Store a result in diskcache with TTL."""
    try:
        _agent_cache.set(key, result, expire=CACHE_TTL)
        logger.debug(f"Cached result for key: {key} (TTL: {CACHE_TTL}s)")
    except Exception as e:
        logger.warning(f"Cache write error for key {key}: {e}")


# ============================================================================
# TOOL: Fetch yfinance data for a ticker
# ============================================================================
def fetch_yfinance_data(ticker: str) -> Dict:
    """
    Tool: Fetch comprehensive financial data for a ticker using yfinance.

    This tool retrieves all available financial data for a ticker and returns
    both full_data (for AI analysis) and display_snapshot (for UI).
    """
    logger.info(f"[Tool] Fetching yfinance data for ticker: {ticker}")
    ticker_upper = ticker.upper()
    cache_key = f"yfinance:{ticker_upper}"

    # Check cache first
    cached = _get_cached_result(cache_key)
    if cached is not None:
        logger.info(f"[Tool] yfinance data for {ticker_upper} loaded from cache")
        return cached

    try:
        logger.debug(f"[Tool] Fetching fresh yfinance data for {ticker_upper}")
        stock = yf.Ticker(ticker_upper)
        info = stock.info

        # Ensure data is JSON-serializable
        try:
            serializable_info = json.loads(json.dumps(info, default=str))
        except Exception as e:
            logger.error(f"[Tool] JSON serialization error for {ticker_upper}: {e}")
            serializable_info = {}

        # Build minimal display snapshot for UI
        display_snapshot = {
            "ticker": ticker_upper,
            "price": serializable_info.get("currentPrice")
            or serializable_info.get("regularMarketPrice"),
            "currency": serializable_info.get("currency", "USD"),
            "pe_ratio": serializable_info.get("trailingPE"),
            "debt_to_equity": serializable_info.get("debtToEquity"),
            "market_cap": serializable_info.get("marketCap"),
            "analyst_ratings": {
                "recommendation": serializable_info.get("recommendationKey"),
                "target_mean": serializable_info.get("targetMeanPrice"),
                "number_of_analysts": serializable_info.get("numberOfAnalystOpinions"),
            },
        }

        payload = {"full_data": serializable_info, "display_snapshot": display_snapshot}

        # Cache the result
        _set_cached_result(cache_key, payload)
        logger.info(
            f"[Tool] yfinance data for {ticker_upper} fetched and cached successfully"
        )
        return payload

    except Exception as e:
        logger.error(f"[Tool] Error fetching yfinance data for {ticker_upper}: {e}")
        raise


# ============================================================================
# TOOL: Fetch news headlines for a ticker
# ============================================================================
def fetch_news_headlines(ticker: str) -> List[Dict]:
    """
    Tool: Fetch recent news headlines for a ticker.

    This tool aggregates news headlines for a ticker.
    Currently mocked; can be replaced with Tavily/Serper API.
    """
    logger.info(f"[Tool] Fetching news headlines for ticker: {ticker}")
    ticker_upper = ticker.upper()
    cache_key = _get_cache_key("NewsAgent", ticker_upper)

    # Check cache first
    cached = _get_cached_result(cache_key)
    if cached is not None:
        logger.info(f"[Tool] News headlines for {ticker_upper} loaded from cache")
        return cached

    try:
        logger.debug(f"[Tool] Generating fresh news headlines for {ticker_upper}")
        # Mock news aggregation (replace with real API)
        headlines = [
            {
                "title": f"{ticker_upper} reports better than expected Q4 results",
                "source": "Finance Daily",
                "sentiment": "positive",
            },
            {
                "title": f"New regulatory scrutiny on {ticker_upper}'s cloud division",
                "source": "Tech Wire",
                "sentiment": "negative",
            },
            {
                "title": f"Analysts raise price targets for {ticker_upper} ahead of keynote",
                "source": "Market Watch",
                "sentiment": "positive",
            },
        ]

        # Cache the result
        _set_cached_result(cache_key, headlines)
        logger.info(
            f"[Tool] News headlines for {ticker_upper} fetched and cached successfully"
        )
        return headlines

    except Exception as e:
        logger.error(f"[Tool] Error fetching news headlines for {ticker_upper}: {e}")
        raise


# ============================================================================
# TOOL: Synthesize investment report using OpenAI
# ============================================================================
def synthesize_investment_report(
    ticker: str, quant_data: Dict, news_data: List[Dict], api_key: str
) -> Dict:
    """
    Tool: Synthesize a comprehensive investment report using OpenAI.

    This tool takes quantitative data and news, sends to OpenAI for analysis,
    and returns a structured investment report.
    """
    logger.info(f"[Tool] Synthesizing investment report for ticker: {ticker}")
    ticker_upper = ticker.upper()
    cache_key = _get_cache_key("Orchestrator", ticker_upper)

    # Check cache first
    cached = _get_cached_result(cache_key)
    if cached is not None:
        logger.info(f"[Tool] Investment report for {ticker_upper} loaded from cache")
        return cached

    try:
        logger.debug(f"[Tool] Calling OpenAI for synthesis of {ticker_upper}")
        client = openai.OpenAI(api_key=api_key)

        prompt = f"""
        You are a Lead Financial Analyst. Synthesize a comprehensive investment report for {ticker_upper}.
        
        You have access to the COMPLETE yfinance dataset below. Analyze ALL available data and focus on 
        what's most important or stands out. Don't limit yourself to specific fields - look at everything 
        including valuation metrics, growth rates, profitability, liquidity, analyst estimates, etc.
        
        Complete Financial Data: {json.dumps(quant_data, default=str)}
        Recent News: {json.dumps(news_data)}
        
        Provide a thorough analysis in JSON format. Use proper formatting for readability:
        - Use \\n\\n for paragraph breaks
        - Use bullet points (•) for lists
        - Use **bold** for emphasis on key metrics
        
        Return these fields (all must be strings):
        1. investment_thesis: Start with a strong opening sentence, then use bullet points to highlight:
           • Key financial strengths or weaknesses
           • Notable valuation insights
           • Growth trajectory
           • Market position
        
        2. risk_reward: Structure as:
           **Upside Potential:**
           • List 2-3 key bullish factors with specific metrics
           
           **Downside Risks:**
           • List 2-3 key bearish factors or concerns
        
        3. portfolio_fit: Structure as:
           **Conservative Investors:** Brief guidance
           **Moderate Investors:** Brief guidance  
           **Aggressive Investors:** Brief guidance
        
        4. bottom_line: A clear, actionable one-sentence recommendation.
        """

        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        report = json.loads(response.choices[0].message.content)
        logger.debug(f"[Tool] OpenAI response parsed successfully for {ticker_upper}")

        # Cache the result
        _set_cached_result(cache_key, report)
        logger.info(
            f"[Tool] Investment report for {ticker_upper} synthesized and cached successfully"
        )
        return report

    except Exception as e:
        logger.error(
            f"[Tool] Error synthesizing investment report for {ticker_upper}: {e}"
        )
        raise


# ============================================================================
# AGENT CLASSES (deprecated, use tools directly)
# ============================================================================
class QuantAgent:
    def __init__(self, ticker_symbol: str):
        self.ticker = ticker_symbol.upper()
        logger.debug(f"QuantAgent initialized for {self.ticker}")

    def get_snapshot(self) -> Dict:
        """Get comprehensive financial data for AI analysis"""
        logger.info(f"QuantAgent.get_snapshot() called for {self.ticker}")
        return fetch_yfinance_data(self.ticker)


class NewsAgent:
    def __init__(self, ticker_symbol: str):
        self.ticker = ticker_symbol.upper()
        logger.debug(f"NewsAgent initialized for {self.ticker}")

    async def get_recent_headlines(self) -> List[Dict]:
        logger.info(f"NewsAgent.get_recent_headlines() called for {self.ticker}")
        return fetch_news_headlines(self.ticker)


class Orchestrator:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        logger.debug("Orchestrator initialized")

    async def synthesize_report(
        self, ticker: str, quant_data: Dict, news_data: List[Dict]
    ) -> Dict:
        logger.info(f"Orchestrator.synthesize_report() called for {ticker}")
        return synthesize_investment_report(
            ticker, quant_data, news_data, self.client.api_key
        )
