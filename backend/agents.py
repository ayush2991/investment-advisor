import os
import yfinance as yf
import openai
from typing import List, Dict, Optional
import json
import logging
import time
import pickle
from pathlib import Path
import diskcache
from urllib.parse import urlparse
import asyncio

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

from agents import Agent, Runner, function_tool

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
            logger.info(f"[Cache] HIT -> {key}")
            return result
        else:
            logger.info(f"[Cache] MISS -> {key}")
            return None
    except (KeyError, pickle.PickleError) as e:
        logger.warning(f"Cache retrieval error for key {key}: {e}")
        return None


def _set_cached_result(key: str, result: Dict) -> None:
    """Store a result in diskcache with TTL."""
    try:
        _agent_cache.set(key, result, expire=CACHE_TTL)
        logger.info(f"[Cache] STORE -> {key} (ttl={CACHE_TTL}s)")
    except (OSError, pickle.PickleError) as e:
        logger.warning(f"Cache write error for key {key}: {e}")


def invalidate_cached_result(key: str) -> bool:
    """Invalidate a specific cached result by key. Returns True if removed."""
    try:
        removed = _agent_cache.delete(key)
        logger.info(f"[Cache] INVALIDATE -> {key} (removed={removed})")
        return bool(removed)
    except (KeyError, OSError) as e:
        logger.warning(f"Cache invalidate error for key {key}: {e}")
        return False


# ============================================================================
# TOOL: Fetch yfinance data for a ticker
# ============================================================================
@function_tool
def fetch_yfinance_data(ticker: str) -> Dict:
    """
    Tool: Fetch comprehensive financial data for a ticker using yfinance.

    This tool retrieves all available financial data for a ticker and returns
    both full_data (for AI analysis) and display_snapshot (for UI).

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
        Dictionary with full_data and display_snapshot
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
        logger.info(f"[Tool] Fetching FRESH yfinance data for {ticker_upper}")
        stock = yf.Ticker(ticker_upper)
        info = stock.info

        # Ensure data is JSON-serializable by converting non-standard types to strings
        try:
            serializable_info = json.loads(json.dumps(info, default=str))
        except (TypeError, ValueError) as e:
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
@function_tool
def fetch_news_headlines(ticker: str) -> List[Dict]:
    """
    Tool: Fetch recent news headlines for a ticker.

    This tool aggregates news headlines for a ticker.
    Uses Tavily if `TAVILY_API_KEY` is configured; otherwise falls back to a mock feed.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
        List of news headline dictionaries with title, source, url, and sentiment
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
        api_key = os.getenv("TAVILY_API_KEY")
        if TavilyClient and api_key:
            logger.info(
                f"[Tool] Using Tavily provider for news headlines for {ticker_upper}"
            )
            client = TavilyClient(api_key=api_key)
            query = f"{ticker_upper} stock news"
            result = client.search(query=query, search_depth="advanced")
            items = result.get("results", [])

            headlines: List[Dict] = []
            for itm in items[:8]:
                url = itm.get("url")
                domain = None
                if url:
                    try:
                        domain = urlparse(url).hostname
                    except (ValueError, AttributeError):
                        domain = None
                headlines.append(
                    {
                        "title": itm.get("title") or itm.get("content") or "",
                        "source": domain or (itm.get("source") or "Web"),
                        "url": url,
                        "sentiment": "neutral",
                    }
                )

            if not headlines:
                logger.info(
                    f"[Tool] Tavily returned empty results for {ticker_upper}; falling back to mock"
                )
                raise RuntimeError("Empty Tavily results")

        else:
            logger.info(
                f"[Tool] TAVILY_API_KEY missing or client unavailable; using mock feed for {ticker_upper}"
            )
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

        _set_cached_result(cache_key, headlines)
        logger.info(
            f"[Tool] News headlines for {ticker_upper} fetched and cached successfully"
        )
        return headlines

    except Exception as e:
        logger.error(f"[Tool] Error fetching news headlines for {ticker_upper}: {e}")
        # As a resilience measure, return a single fallback item instead of failing the pipeline
        fallback = [
            {
                "title": f"Latest market headlines for {ticker_upper} are unavailable right now.",
                "source": "System",
                "sentiment": "neutral",
            }
        ]
        return fallback


def invalidate_news_cache(ticker: str) -> bool:
    """Invalidate cached news headlines for a ticker."""
    ticker_upper = ticker.upper()
    cache_key = _get_cache_key("NewsAgent", ticker_upper)
    return invalidate_cached_result(cache_key)


# ============================================================================
# AGENT: Data Collection Agent - Gathers financial and news data
# ============================================================================
data_collector_agent = Agent(
    name="Data Collector",
    instructions="""You are a financial data collection agent. Your job is to gather comprehensive 
financial and news information about a stock ticker. Use the available tools to fetch:
1. Detailed financial data using fetch_yfinance_data()
2. Recent news headlines using fetch_news_headlines()

Return the collected data in a structured format for analysis.""",
    tools=[fetch_yfinance_data, fetch_news_headlines],
)


# ============================================================================
# AGENT: Investment Analyst Agent - Analyzes data and provides recommendations
# ============================================================================
analyst_agent = Agent(
    name="Investment Analyst",
    instructions="""You are a Lead Financial Analyst specializing in investment recommendations. 
Your role is to synthesize comprehensive investment reports based on financial data and market news.

When analyzing a stock, consider:
- Valuation metrics (P/E ratio, Price-to-Book, etc.)
- Financial health (debt-to-equity, cash flow, profitability)
- Growth trajectory and market position
- Recent news sentiment and catalysts
- Risk factors and opportunities

Provide thorough analysis structured as:
1. Investment Thesis: Opening statement + key bullet points on strengths/weaknesses
2. Risk/Reward: Clear upside potential and downside risks with specific metrics
3. Portfolio Fit: How it aligns with different investor types (Conservative/Moderate/Aggressive)
4. Bottom Line: Clear, actionable one-sentence recommendation

Format all analysis as JSON with these fields: investment_thesis, risk_reward, portfolio_fit, bottom_line.
Use proper formatting in string values (\\n\\n for breaks, bullet points, **bold** for emphasis).""",
)


# ============================================================================
# ORCHESTRATOR FUNCTIONS
# ============================================================================


async def collect_investment_data(ticker: str) -> Dict:
    """
    Collect financial and news data for a ticker using the Data Collector agent.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary containing financial and news data
    """
    logger.info(f"[Orchestrator] Collecting data for {ticker}")
    prompt = f"Collect all available financial and news data for the ticker {ticker}"

    result = await Runner.run(data_collector_agent, prompt)
    logger.info(f"[Orchestrator] Data collection complete for {ticker}")
    return result.final_output


async def analyze_investment(
    ticker: str, financial_data: Dict, news_data: List[Dict]
) -> Dict:
    """
    Analyze investment opportunity using the Investment Analyst agent.

    Args:
        ticker: Stock ticker symbol
        financial_data: Complete financial dataset from yfinance
        news_data: List of recent news headlines

    Returns:
        Investment analysis report
    """
    logger.info(f"[Orchestrator] Analyzing investment for {ticker}")

    prompt = f"""Analyze {ticker} and provide a comprehensive investment report.

Financial Data: {json.dumps(financial_data, default=str)}

Recent News: {json.dumps(news_data)}

Return a detailed JSON analysis with investment_thesis, risk_reward, portfolio_fit, and bottom_line."""

    result = await Runner.run(analyst_agent, prompt)
    logger.info(f"[Orchestrator] Analysis complete for {ticker}")

    # Parse the response if it's JSON, otherwise wrap it in structured format
    try:
        if isinstance(result.final_output, dict):
            return result.final_output
        else:
            return json.loads(result.final_output)
    except (json.JSONDecodeError, TypeError):
        # If parsing fails, return wrapped response with empty analysis sections
        return {
            "investment_thesis": result.final_output,
            "risk_reward": "",
            "portfolio_fit": "",
            "bottom_line": "",
        }


# Maintain backward compatibility for direct tool access
async def synthesize_investment_report(
    ticker: str, quant_data: Dict, news_data: List[Dict], api_key: str = None
) -> Dict:
    """
    Backward-compatible wrapper: Synthesize investment report.

    This function is maintained for backward compatibility with the FastAPI endpoints.
    It uses the Investment Analyst agent to generate the report.
    """
    return await analyze_investment(ticker, quant_data, news_data)
