from __future__ import annotations

import os
import asyncio
from datetime import datetime, timedelta
import hashlib
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from urllib.parse import urlparse
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from agents import Agent, Runner, AgentOutputSchema
from cachetools import TTLCache
from pydantic import BaseModel

APP_NAME = "PortfolioPulse Advisor"
TAVILY_ENDPOINT = "https://api.tavily.com/search"

app = FastAPI(title=APP_NAME)
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("atlas.ai")

INFO_CACHE = TTLCache(maxsize=256, ttl=1800)
HISTORY_CACHE = TTLCache(maxsize=256, ttl=1800)
NEWS_CACHE = TTLCache(maxsize=256, ttl=900)
AI_CACHE = TTLCache(maxsize=256, ttl=1800)
PORTFOLIO_CACHE = TTLCache(maxsize=256, ttl=1800)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount(
    "/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static"
)


def _safe_float(value):
    try:
        if value is None:
            return None
        value = float(value)
        if np.isnan(value):
            return None
        return value
    except Exception:
        return None


def _pct_change(series: pd.Series) -> float | None:
    if series is None or len(series) < 2:
        return None
    return (series.iloc[-1] / series.iloc[0] - 1) * 100


def _calc_max_drawdown(prices: pd.Series, window: int | None = None) -> float | None:
    if prices is None or prices.empty:
        return None
    if window and window > 1:
        running_max = prices.rolling(window=window, min_periods=1).max()
    else:
        running_max = prices.cummax()
    drawdown = (prices / running_max - 1) * 100
    return float(drawdown.min())


def _calc_volatility(returns: pd.Series, trading_days: int) -> float | None:
    if returns is None or returns.empty:
        return None
    return float(returns.std() * np.sqrt(trading_days) * 100)


def _trend_label(short_ma: float | None, long_ma: float | None) -> str:
    if short_ma is None or long_ma is None:
        return "Unknown"
    if short_ma > long_ma:
        return "Uptrend"
    if short_ma < long_ma:
        return "Downtrend"
    return "Sideways"


def _format_money(value: float | None) -> str | None:
    if value is None:
        return None
    abs_val = abs(value)
    if abs_val >= 1e12:
        return f"${value/1e12:.2f}T"
    if abs_val >= 1e9:
        return f"${value/1e9:.2f}B"
    if abs_val >= 1e6:
        return f"${value/1e6:.2f}M"
    return f"${value:,.0f}"


def _get_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
        if netloc.startswith("www."):
            return netloc[4:]
        return netloc
    except Exception:
        return ""


def _sanitize_for_json(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(k): _sanitize_for_json(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_sanitize_for_json(v) for v in payload]
    if isinstance(payload, (np.integer, np.int64)):
        return int(payload)
    if isinstance(payload, (np.floating, np.float64)):
        return None if np.isnan(payload) else float(payload)
    if isinstance(payload, (np.bool_,)):
        return bool(payload)
    if isinstance(payload, (pd.Timestamp, datetime)):
        return payload.isoformat()
    return payload


def _safe_dump(result) -> dict | None:
    if result and getattr(result, "final_output", None):
        return result.final_output.model_dump()
    return None


def _cache_get(cache: TTLCache, key: str):
    try:
        return cache.get(key)
    except Exception:
        return None


def _cache_set(cache: TTLCache, key: str, value):
    try:
        cache[key] = value
    except Exception:
        return None


def _hash_payload(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_etf(info: dict) -> bool:
    quote_type = (info.get("quoteType") or "").upper()
    if quote_type in {"ETF", "MUTUALFUND", "FUND"}:
        return True
    if info.get("fundFamily") or info.get("category"):
        return True
    return False


def _normalize_weights(holdings: list[PortfolioInput]) -> list[PortfolioInput]:
    total = sum(h.weight for h in holdings if h.weight is not None)
    if total == 0:
        return holdings
    return [PortfolioInput(ticker=h.ticker, weight=h.weight / total) for h in holdings]


def _portfolio_drawdown(portfolio_values: pd.Series) -> float | None:
    if portfolio_values is None or portfolio_values.empty:
        return None
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values / running_max - 1) * 100
    return float(drawdown.min())


def _annualized_return(
    returns: pd.Series, trading_days: int, method: str
) -> float | None:
    if returns is None or returns.empty:
        return None
    if method == "log":
        return float((np.exp(returns.mean() * trading_days) - 1) * 100)
    return float((1 + returns.mean()) ** trading_days - 1) * 100


def _annualized_volatility(returns: pd.Series, trading_days: int) -> float | None:
    if returns is None or returns.empty:
        return None
    return float(returns.std() * np.sqrt(trading_days) * 100)


def _strip_tz(index: pd.Index) -> pd.Index:
    if hasattr(index, "tz") and index.tz is not None:
        return index.tz_localize(None)
    return index


def _interval_days(interval: str) -> int:
    return {"1d": 1, "1wk": 5, "1mo": 21}.get(interval, 1)


def _window_periods(days: int, interval: str) -> int:
    per = _interval_days(interval)
    return max(2, int(days / per))


def _series_returns(prices: pd.Series, method: str) -> pd.Series:
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def _sharpe_ratio(
    annual_return: float | None, annual_volatility: float | None, risk_free: float
) -> float | None:
    if annual_return is None or annual_volatility in (None, 0):
        return None
    return (annual_return - risk_free) / annual_volatility


class FundamentalsBrief(BaseModel):
    summary: str
    positives: list[str]
    concerns: list[str]
    analyst_view: str | None = None
    earnings: str | None = None


class TechnicalBrief(BaseModel):
    summary: str
    trend: str
    momentum_notes: list[str]
    risk_notes: list[str]


class NewsBrief(BaseModel):
    summary: str
    key_items: list[str]
    sentiment: str | None = None


class StockBrief(BaseModel):
    summary: str
    upsides: list[str]
    risks: list[str]
    watch_items: list[str]
    analyst_view: str | None = None
    earnings: str | None = None


def _model_name(override: str | None = None) -> str:
    if override:
        return override
    # Use gpt-4o-mini as a reliable default fallback
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class PortfolioInput(BaseModel):
    ticker: str
    weight: float


class PortfolioSummary(BaseModel):
    summary: str = ""
    current_portfolio_analysis: str = ""
    suggested_portfolio_analysis: str = ""
    strengths: list[str] = []
    weaknesses: list[str] = []
    improvements: list[str] = []
    suggested_portfolios: list[list[PortfolioInput]] = []
    comparison: str = ""


class NewsQuery(BaseModel):
    queries: list[str]


class NewsSelection(BaseModel):
    selected_indices: list[int]


class ComparisonBrief(BaseModel):
    summary: str = ""
    portfolio_a_analysis: str = ""
    portfolio_b_analysis: str = ""
    comparison_table_summary: str = ""
    strengths_a: list[str] = []
    strengths_b: list[str] = []
    weaknesses_a: list[str] = []
    weaknesses_b: list[str] = []
    best_for_scenarios: list[dict[str, str]] = []
    verdict: str = ""


def _build_fundamentals_agent(model: str | None = None) -> Agent:
    return Agent(
        name="Fundamentals Agent",
        instructions=(
            "Analyze fundamentals using only the provided ticker_info JSON. "
            "Focus on valuation, growth, profitability, balance sheet hints, and analyst ratings if present. "
            "If the instrument is an ETF or fund, describe it accordingly and do not apply single-stock "
            "valuation logic unless the data supports it. If data is missing, note that explicitly. "
            "Avoid speculation."
        ),
        model=_model_name(model),
        output_type=AgentOutputSchema(FundamentalsBrief, strict_json_schema=False),
    )


def _build_technicals_agent(model: str | None = None) -> Agent:
    return Agent(
        name="Technicals Agent",
        instructions=(
            "Analyze technical and price-based signals using only the provided technicals JSON. "
            "Summarize trend, momentum, and notable risk signals. Avoid speculation."
        ),
        model=_model_name(model),
        output_type=AgentOutputSchema(TechnicalBrief, strict_json_schema=False),
    )


def _build_news_agent(model: str | None = None) -> Agent:
    return Agent(
        name="News Agent",
        instructions=(
            "Summarize the provided news list into key items and overall sentiment. "
            "Only use the news payload; if none is present, say so."
        ),
        model=_model_name(model),
        output_type=AgentOutputSchema(NewsBrief, strict_json_schema=False),
    )


def _build_synthesis_agent(model: str | None = None) -> Agent:
    return Agent(
        name="Synthesis Agent",
        instructions=(
            "Combine the fundamentals, technicals, and news summaries into a concise, "
            "client-ready brief. Only use the provided JSON. If data is missing, say so."
        ),
        model=_model_name(model),
        output_type=AgentOutputSchema(StockBrief, strict_json_schema=False),
    )


def _build_portfolio_agent(model: str | None = None) -> Agent:
    return Agent(
        name="Portfolio Analyst",
        instructions=(
            "You are a portfolio analyst. Use only the provided JSON to summarize portfolio-level "
            "strengths, weaknesses, and improvements. "
            "You MUST provide all fields in the output schema: "
            "1. summary: High-level overview of the current state. "
            "2. current_portfolio_analysis: Brief qualitative analysis of the current holdings. "
            "3. suggested_portfolio_analysis: Brief qualitative analysis of why the suggested alternative is better. "
            "4. strengths: Core strengths of the current setup. "
            "5. weaknesses: Core risks or weaknesses of the current setup. "
            "6. improvements: Specific tactical steps to improve the portfolio. "
            "7. suggested_portfolios: A list containing exactly one list of PortfolioInput objects (ticker/weight) that sum to 1.0 (100%). "
            "8. comparison: A final verdict comparing the current vs the suggested state. "
            "Avoid speculation."
        ),
        model=_model_name(model),
        output_type=AgentOutputSchema(PortfolioSummary, strict_json_schema=False),
    )


def _build_news_query_agent(model: str | None = None) -> Agent:
    return Agent(
        name="News Query Agent",
        instructions=(
            "Generate 3-5 high-quality, diverse search queries to find the most relevant and recent "
            "financial news and market analysis for the given stock ticker and company name. "
            "Focus on different aspects like recent earnings, product launches, analyst upgrades/downgrades, "
            "regulatory news, and industry trends. Output as a JSON list of strings."
        ),
        model=_model_name(model),
        output_type=AgentOutputSchema(NewsQuery, strict_json_schema=False),
    )


def _build_news_selection_agent(model: str | None = None) -> Agent:
    return Agent(
        name="News Selection Agent",
        instructions=(
            "From the provided list of news articles, pick the 5-7 most relevant and high-quality "
            "articles for an investor. Focus on primary sources, detailed analysis, and recent events. "
            "Avoid duplicates and low-quality summaries. Return a list of indices (0-indexed) of the "
            "selected articles in order of relevance."
        ),
        model=_model_name(model),
        output_type=AgentOutputSchema(NewsSelection, strict_json_schema=False),
    )


def _build_comparison_agent(model: str | None = None) -> Agent:
    return Agent(
        name="Portfolio Comparison Agent",
        instructions=(
            "You are a senior investment strategist. Compare two investment portfolios (A and B) based on "
            "their risk/return stats, sector allocations, and holdings. "
            "You MUST provide all fields in the output schema: "
            "1. summary: High-level tactical overview. "
            "2. portfolio_a_analysis: Deep dive into A. "
            "3. portfolio_b_analysis: Deep dive into B. "
            "4. comparison_table_summary: Brief text summarizing the key statistical differences. "
            "5. strengths_a/b & weaknesses_a/b: Detailed lists for both. "
            "6. best_for_scenarios: A list of dicts with 'scenario' and 'analysis' keys. "
            "7. verdict: Final conclusion on which is better for whom. "
            "Use only provided data. Do not skip any fields."
        ),
        model=_model_name(model),
        output_type=AgentOutputSchema(ComparisonBrief, strict_json_schema=False),
    )


async def _openai_stock_brief(
    info: dict, technicals: dict, news: list[dict], model: str | None = None
) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "status": "unavailable",
            "message": "Set OPENAI_API_KEY to enable AI-generated analysis.",
        }

    sanitized_info = _sanitize_for_json(info)
    sanitized_technicals = _sanitize_for_json(technicals)
    sanitized_news = _sanitize_for_json(news)

    try:
        cache_key = _hash_payload(
            {
                "info": sanitized_info,
                "technicals": sanitized_technicals,
                "news": sanitized_news,
                "model": _model_name(model),
            }
        )
        cached = _cache_get(AI_CACHE, cache_key)
        if cached is not None:
            logger.debug("ai summary cache hit for %s", cache_key)
            return cached

        fundamentals_agent = _build_fundamentals_agent(model)
        technicals_agent = _build_technicals_agent(model)
        news_agent = _build_news_agent(model)
        synthesis_agent = _build_synthesis_agent(model)

        logger.debug(
            "fundamentals input: %s",
            json.dumps({"ticker_info": sanitized_info}, default=str),
        )
        fundamentals_result = await Runner.run(
            fundamentals_agent, json.dumps({"ticker_info": sanitized_info})
        )
        logger.debug("fundamentals output: %s", _safe_dump(fundamentals_result))
        logger.debug(
            "technicals input: %s",
            json.dumps({"technicals": sanitized_technicals}, default=str),
        )
        technicals_result = await Runner.run(
            technicals_agent, json.dumps({"technicals": sanitized_technicals})
        )
        logger.debug("technicals output: %s", _safe_dump(technicals_result))
        logger.debug(
            "news input: %s", json.dumps({"news": sanitized_news}, default=str)
        )
        news_result = await Runner.run(news_agent, json.dumps({"news": sanitized_news}))
        logger.debug("news output: %s", _safe_dump(news_result))

        synthesis_payload = {
            "ticker_info": sanitized_info,
            "technicals": sanitized_technicals,
            "news": sanitized_news,
            "fundamentals_summary": _safe_dump(fundamentals_result),
            "technicals_summary": _safe_dump(technicals_result),
            "news_summary": _safe_dump(news_result),
        }
        logger.debug("synthesis input: %s", json.dumps(synthesis_payload, default=str))
        result = await Runner.run(synthesis_agent, json.dumps(synthesis_payload))
    except Exception:
        logger.exception("AI analysis failed for stock brief")
        return {"status": "error", "message": "AI analysis failed to run."}

    final_output = result.final_output
    if not final_output:
        return {"status": "error", "message": "AI analysis returned no content."}

    parsed = final_output.model_dump()
    parsed["status"] = "ok"
    parsed["agent_outputs"] = {
        "fundamentals": _safe_dump(fundamentals_result),
        "technicals": _safe_dump(technicals_result),
        "news": _safe_dump(news_result),
    }
    logger.debug("synthesis output: %s", parsed)
    _cache_set(AI_CACHE, cache_key, parsed)
    return parsed


async def _openai_portfolio_brief(payload: dict, model: str | None = None) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "status": "unavailable",
            "message": "Set OPENAI_API_KEY to enable AI-generated analysis.",
        }

    cache_key = _hash_payload({**payload, "model": _model_name(model)})
    cached = _cache_get(AI_CACHE, cache_key)
    if cached is not None:
        logger.debug("portfolio ai cache hit for %s", cache_key)
        return cached

    agent = _build_portfolio_agent(model)
    try:
        logger.debug("portfolio agent input: %s", json.dumps(payload, default=str))
        result = await Runner.run(agent, json.dumps(payload))
    except Exception:
        logger.exception("AI analysis failed for portfolio brief")
        return {"status": "error", "message": "AI analysis failed to run."}

    if not result.final_output:
        return {"status": "error", "message": "AI analysis returned no content."}

    parsed = result.final_output.model_dump()
    parsed["status"] = "ok"
    logger.debug("portfolio agent output: %s", parsed)
    _cache_set(AI_CACHE, cache_key, parsed)
    return parsed


async def _openai_comparison_brief(payload: dict, model: str | None = None) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "status": "unavailable",
            "message": "Set OPENAI_API_KEY to enable AI-generated analysis.",
        }

    cache_key = _hash_payload(
        {**payload, "type": "comparison", "model": _model_name(model)}
    )
    cached = _cache_get(AI_CACHE, cache_key)
    if cached is not None:
        logger.debug("comparison ai cache hit for %s", cache_key)
        return cached

    agent = _build_comparison_agent(model)
    try:
        logger.debug("comparison agent input: %s", json.dumps(payload, default=str))
        result = await Runner.run(agent, json.dumps(payload))
    except Exception:
        logger.exception("AI analysis failed for comparison brief")
        return {"status": "error", "message": "AI analysis failed to run."}

    if not result.final_output:
        return {"status": "error", "message": "AI analysis returned no content."}

    parsed = result.final_output.model_dump()
    parsed["status"] = "ok"
    logger.debug("comparison agent output: %s", parsed)
    _cache_set(AI_CACHE, cache_key, parsed)
    return parsed


async def _fetch_news(
    ticker: str, company_name: str | None, model: str | None = None
) -> list[dict]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set, news fetch will be skipped.")
        return []

    cache_key = f"tavily_news_v2:{ticker}:{company_name or ''}:{_model_name(model)}"
    cached = _cache_get(NEWS_CACHE, cache_key)
    if cached is not None:
        logger.debug("news cache hit for %s", cache_key)
        return cached

    # 1. Generate queries
    query_agent = _build_news_query_agent(model)
    try:
        query_input = f"Ticker: {ticker}, Company: {company_name or 'Unknown'}"
        query_result = await Runner.run(query_agent, query_input)
        queries = (
            query_result.final_output.queries
            if query_result and query_result.final_output
            else [f"{ticker} stock recent news"]
        )
    except Exception:
        logger.exception("failed to generate news queries for %s", ticker)
        queries = [f"{ticker} latest stock market news".strip()]

    all_results = []
    seen_urls = set()

    # 2. Run searches using Tavily
    for query in queries[:2]:  # Limit to 2 queries to be efficient
        try:

            def _post():
                three_days_ago = (datetime.utcnow() - timedelta(days=3)).strftime(
                    "%Y-%m-%d"
                )
                return requests.post(
                    TAVILY_ENDPOINT,
                    json={
                        "api_key": api_key,
                        "query": query,
                        "search_depth": "advanced",
                        "topic": "news",
                        "start_date": three_days_ago,
                        "max_results": 10,
                    },
                    timeout=15,
                )

            response = await asyncio.to_thread(_post)
            response.raise_for_status()
            payload = response.json()
            for item in payload.get("results", []):
                url = item.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(
                        {
                            "title": item.get("title"),
                            "url": url,
                            "description": item.get("content"),
                            "age": item.get(
                                "published_date"
                            ),  # Tavily returns ISO dates for news
                            "domain": _get_domain(url),
                        }
                    )
        except Exception:
            logger.exception("tavily news fetch failed for query: %s", query)

    if not all_results:
        return []

    # 3. Rank/Select
    selection_agent = _build_news_selection_agent(model)
    try:
        selection_input = json.dumps(
            [
                {
                    "index": i,
                    "title": r["title"],
                    "description": (r["description"] or "")[:200],
                }
                for i, r in enumerate(all_results)
            ]
        )
        selection_result = await Runner.run(selection_agent, selection_input)
        if (
            selection_result
            and selection_result.final_output
            and selection_result.final_output.selected_indices
        ):
            indices = selection_result.final_output.selected_indices
            final_results = [all_results[i] for i in indices if i < len(all_results)]
        else:
            final_results = all_results[:7]
    except Exception:
        logger.exception("failed to select news articles for %s", ticker)
        final_results = all_results[:7]

    _cache_set(NEWS_CACHE, cache_key, final_results)
    return final_results


def _fetch_info_cached(ticker: str) -> dict:
    info_cache_key = f"info:{ticker}"
    cached_info = _cache_get(INFO_CACHE, info_cache_key)
    if cached_info is not None:
        logger.debug("info cache hit for %s", info_cache_key)
        return cached_info
    yf_ticker = yf.Ticker(ticker)
    try:
        info = yf_ticker.info or {}
    except Exception:
        logger.exception("yfinance info fetch failed for %s", ticker)
        info = {}
    _cache_set(INFO_CACHE, info_cache_key, info)
    return info


def _portfolio_stats(
    holdings: list[PortfolioInput],
    start_date: datetime.date,
    end_date: datetime.date,
    trading_days: int,
    risk_free_rate: float,
    price_interval: str,
    auto_adjust: bool,
    benchmark_ticker: str,
    return_method: str,
    drawdown_method: str,
    drawdown_window_days: int,
    beta_lookback_days: int,
    risk_free_source: str,
    risk_free_ticker: str,
) -> dict:
    tickers = [h.ticker for h in holdings]
    if not tickers:
        return {"error": "No tickers provided."}

    history_cache_key = f"portfolio:{','.join(tickers)}:{start_date}:{end_date}:{price_interval}:{auto_adjust}"
    cached = _cache_get(HISTORY_CACHE, history_cache_key)
    if cached is not None:
        logger.debug("portfolio history cache hit for %s", history_cache_key)
        price_data = cached
    else:
        try:
            price_data = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                interval=price_interval,
                auto_adjust=auto_adjust,
                progress=False,
                group_by="ticker",
            )
        except Exception:
            logger.exception("yfinance download failed for portfolio")
            return {"error": "Unable to fetch price data."}
        _cache_set(HISTORY_CACHE, history_cache_key, price_data)

    if price_data is None or price_data.empty:
        return {"error": "No historical data found."}

    if isinstance(price_data.columns, pd.MultiIndex):
        close_prices = pd.DataFrame(
            {
                ticker: price_data[ticker]["Close"]
                for ticker in tickers
                if ticker in price_data
            }
        )
    else:
        close_prices = price_data["Close"].to_frame()

    close_prices = close_prices.dropna(how="all")
    returns = close_prices.apply(lambda s: _series_returns(s, return_method))
    returns = returns.dropna(how="all")

    weights = np.array([h.weight for h in holdings], dtype=float)
    weight_map = {h.ticker: h.weight for h in holdings}
    aligned_weights = np.array(
        [weight_map.get(ticker, 0) for ticker in close_prices.columns], dtype=float
    )

    portfolio_returns = (returns * aligned_weights).sum(axis=1)
    if return_method == "log":
        portfolio_values = np.exp(portfolio_returns.cumsum())
    else:
        portfolio_values = (1 + portfolio_returns).cumprod()

    drawdown_window = (
        _window_periods(drawdown_window_days, price_interval)
        if drawdown_method == "rolling"
        else None
    )
    stats = {
        "annualized_return": _annualized_return(
            portfolio_returns, trading_days, return_method
        ),
        "annualized_volatility": _annualized_volatility(
            portfolio_returns, trading_days
        ),
        "max_drawdown": _calc_max_drawdown(portfolio_values, drawdown_window),
    }

    sector_alloc = {}
    holding_details = []
    for holding in holdings:
        info = _fetch_info_cached(holding.ticker)
        is_etf = _is_etf(info)
        sector = info.get("sector") or ("ETF / Fund" if is_etf else "Unknown")
        sector_alloc[sector] = sector_alloc.get(sector, 0) + holding.weight
        holding_details.append(
            {
                "ticker": holding.ticker,
                "weight": holding.weight,
                "company_name": info.get("shortName") or info.get("longName"),
                "sector": sector,
                "industry": info.get("industry"),
                "market_cap": _safe_float(info.get("marketCap")),
                "beta": _safe_float(info.get("beta")),
                "is_etf": is_etf,
                "fund_category": info.get("category"),
                "fund_family": info.get("fundFamily"),
                "fund_holdings": info.get("holdings") or info.get("fundHoldings"),
            }
        )

    hhi = float(np.sum(np.square(aligned_weights)))
    effective_holdings = float(1 / hhi) if hhi > 0 else None

    benchmark = benchmark_ticker.upper().strip() if benchmark_ticker else "SPY"
    spy_info = _fetch_info_cached(benchmark)
    beta = None
    try:
        spy_hist = yf.Ticker(benchmark).history(
            start=start_date,
            end=end_date,
            interval=price_interval,
            auto_adjust=auto_adjust,
        )
        spy_returns = _series_returns(spy_hist["Close"], return_method)
        portfolio_returns.index = _strip_tz(portfolio_returns.index)
        spy_returns.index = _strip_tz(spy_returns.index)
        joined = pd.concat([portfolio_returns, spy_returns], axis=1).dropna()
        if beta_lookback_days:
            window = _window_periods(beta_lookback_days, price_interval)
            joined = joined.iloc[-window:]
        if not joined.empty:
            cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1]
            var = np.var(joined.iloc[:, 1])
            if var != 0:
                beta = float(cov / var)
    except Exception:
        logger.exception("benchmark beta calculation failed")

    risk_free = float(risk_free_rate)
    if risk_free_source == "ticker" and risk_free_ticker:
        try:
            rf_hist = yf.Ticker(risk_free_ticker).history(
                start=start_date,
                end=end_date,
                interval=price_interval,
                auto_adjust=auto_adjust,
            )
            rf_series = rf_hist["Close"].dropna()
            if not rf_series.empty:
                risk_free = float(rf_series.mean() / 100)
        except Exception:
            logger.exception("risk-free ticker fetch failed")
    risk_free_percent = risk_free * 100
    stats["beta"] = beta or _safe_float(spy_info.get("beta"))
    stats["concentration_hhi"] = hhi
    stats["effective_holdings"] = effective_holdings
    stats["sharpe_ratio"] = _sharpe_ratio(
        stats["annualized_return"],
        stats["annualized_volatility"],
        risk_free_percent,
    )
    stats["risk_free_rate"] = risk_free_percent

    return {
        "stats": stats,
        "sector_allocation": sector_alloc,
        "holdings": holding_details,
    }


def _compare_portfolios(user_stats: dict, suggested_stats: dict) -> str:
    if not user_stats or not suggested_stats:
        return ""
    u = user_stats.get("stats", {})
    s = suggested_stats.get("stats", {})

    def _delta(a, b):
        if a is None or b is None:
            return None
        return b - a

    ret = _delta(u.get("annualized_return"), s.get("annualized_return"))
    sharpe = _delta(u.get("sharpe_ratio"), s.get("sharpe_ratio"))
    vol = _delta(u.get("annualized_volatility"), s.get("annualized_volatility"))
    dd = _delta(u.get("max_drawdown"), s.get("max_drawdown"))

    narrative = []
    if ret is not None:
        if ret > 0:
            narrative.append(
                "The suggested portfolio targets higher expected return, which can suit growth-oriented goals."
            )
        else:
            narrative.append(
                "The suggested portfolio targets lower expected return, which may fit more defensive objectives."
            )
    if sharpe is not None:
        if sharpe > 0:
            narrative.append(
                "Its Sharpe ratio is stronger, indicating better risk-adjusted performance in the same period."
            )
        else:
            narrative.append(
                "Its Sharpe ratio is weaker, suggesting the current portfolio delivers better risk-adjusted returns."
            )
    if vol is not None:
        if vol > 0:
            narrative.append(
                "Volatility is higher, so performance may swing more and requires higher risk tolerance."
            )
        else:
            narrative.append(
                "Volatility is lower, which can be preferable for steadier profiles or shorter horizons."
            )
    if dd is not None:
        if dd > 0:
            narrative.append(
                "Drawdowns are less severe, making it potentially better for capital preservation."
            )
        else:
            narrative.append(
                "Drawdowns are deeper, so the current portfolio may be more resilient in stress periods."
            )

    return " ".join(narrative)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "app_name": APP_NAME}
    )


@app.get("/api/analyze")
async def analyze_stock(
    ticker: str,
    lookback_days: int = 365,
    trading_days: int = 252,
    short_ma_window: int = 50,
    long_ma_window: int = 200,
    price_interval: str = "1d",
    auto_adjust: bool = True,
    return_method: str = "pct",
    drawdown_method: str = "full",
    drawdown_window_days: int = 365,
    model: str | None = None,
):
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required.")

    ticker = ticker.upper().strip()
    yf_ticker = yf.Ticker(ticker)

    info_cache_key = f"info:{ticker}"
    cached_info = _cache_get(INFO_CACHE, info_cache_key)
    if cached_info is not None:
        logger.debug("info cache hit for %s", info_cache_key)
        info = cached_info
    else:
        try:
            info = yf_ticker.info or {}
        except Exception:
            logger.exception("yfinance info fetch failed for %s", ticker)
            info = {}
        _cache_set(INFO_CACHE, info_cache_key, info)

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=lookback_days)

    history_cache_key = (
        f"hist:{ticker}:{start_date}:{end_date}:{price_interval}:{auto_adjust}"
    )
    cached_history = _cache_get(HISTORY_CACHE, history_cache_key)
    if cached_history is not None:
        logger.debug("history cache hit for %s", history_cache_key)
        history = cached_history
    else:
        try:
            history = yf_ticker.history(
                start=start_date,
                end=end_date,
                interval=price_interval,
                auto_adjust=auto_adjust,
            )
        except Exception:
            logger.exception("yfinance history fetch failed for %s", ticker)
            history = pd.DataFrame()
        _cache_set(HISTORY_CACHE, history_cache_key, history)

    if history.empty:
        raise HTTPException(status_code=404, detail="No historical data found.")

    history = history.dropna(subset=["Close"])
    close_prices = history["Close"]
    returns = _series_returns(close_prices, return_method)

    last_price = float(close_prices.iloc[-1])
    price_change_1y = _pct_change(close_prices)
    volatility = _calc_volatility(returns, trading_days)
    drawdown_window = (
        _window_periods(drawdown_window_days, price_interval)
        if drawdown_method == "rolling"
        else None
    )
    max_drawdown = _calc_max_drawdown(close_prices, drawdown_window)

    short_ma = _safe_float(close_prices.rolling(window=short_ma_window).mean().iloc[-1])
    long_ma = _safe_float(close_prices.rolling(window=long_ma_window).mean().iloc[-1])

    fifty_two_week_low = _safe_float(info.get("fiftyTwoWeekLow"))
    fifty_two_week_high = _safe_float(info.get("fiftyTwoWeekHigh"))
    price_vs_52w = None
    if fifty_two_week_low is not None and fifty_two_week_high is not None:
        if fifty_two_week_high != fifty_two_week_low:
            price_vs_52w = (
                (last_price - fifty_two_week_low)
                / (fifty_two_week_high - fifty_two_week_low)
                * 100
            )

    metrics = {
        "company_name": info.get("shortName") or info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": _safe_float(info.get("marketCap")),
        "pe_ratio": _safe_float(info.get("trailingPE")),
        "forward_pe": _safe_float(info.get("forwardPE")),
        "dividend_yield": _safe_float(info.get("dividendYield")),
        "revenue_growth": _safe_float(info.get("revenueGrowth")),
        "gross_margins": _safe_float(info.get("grossMargins")),
        "beta": _safe_float(info.get("beta")),
        "price": last_price,
        "price_change_1y": price_change_1y,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "trend": _trend_label(short_ma, long_ma),
        "fifty_two_week_low": fifty_two_week_low,
        "fifty_two_week_high": fifty_two_week_high,
        "price_vs_52w": price_vs_52w,
    }

    news = await _fetch_news(ticker, metrics.get("company_name"), model=model)
    technicals = {
        "price_change_1y": price_change_1y,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "trend": _trend_label(short_ma, long_ma),
        "fifty_two_week_low": fifty_two_week_low,
        "fifty_two_week_high": fifty_two_week_high,
        "price_vs_52w": price_vs_52w,
        "lookback_days": lookback_days,
        "trading_days": trading_days,
        "short_ma_window": short_ma_window,
        "long_ma_window": long_ma_window,
        "price_interval": price_interval,
        "auto_adjust": auto_adjust,
        "return_method": return_method,
        "drawdown_method": drawdown_method,
        "drawdown_window_days": drawdown_window_days,
    }
    ai_summary = await _openai_stock_brief(info, technicals, news, model=model)

    response = {
        "ticker": ticker,
        "as_of": end_date.isoformat(),
        "metrics": {
            "company_name": metrics.get("company_name"),
            "sector": metrics.get("sector"),
            "industry": metrics.get("industry"),
            "market_cap": _format_money(metrics.get("market_cap")),
            "pe_ratio": metrics.get("pe_ratio"),
            "forward_pe": metrics.get("forward_pe"),
            "dividend_yield": metrics.get("dividend_yield"),
            "revenue_growth": metrics.get("revenue_growth"),
            "gross_margins": metrics.get("gross_margins"),
            "beta": metrics.get("beta"),
            "price": metrics.get("price"),
            "price_change_1y": metrics.get("price_change_1y"),
            "volatility": metrics.get("volatility"),
            "max_drawdown": metrics.get("max_drawdown"),
            "trend": metrics.get("trend"),
            "fifty_two_week_low": metrics.get("fifty_two_week_low"),
            "fifty_two_week_high": metrics.get("fifty_two_week_high"),
            "price_vs_52w": metrics.get("price_vs_52w"),
        },
        "ai_summary": ai_summary,
        "news": news,
        "disclaimer": "This is for educational purposes only and not investment advice.",
        "assumptions": {
            "lookback_days": lookback_days,
            "trading_days": trading_days,
            "short_ma_window": short_ma_window,
            "long_ma_window": long_ma_window,
            "price_interval": price_interval,
            "auto_adjust": auto_adjust,
            "return_method": return_method,
            "drawdown_method": drawdown_method,
            "drawdown_window_days": drawdown_window_days,
            "beta_lookback_days": None,
            "risk_free_rate": None,
            "risk_free_source": None,
            "risk_free_ticker": None,
            "benchmark_ticker": None,
        },
    }

    return response


class PortfolioRequest(BaseModel):
    holdings: list[PortfolioInput]
    lookback_days: int = 365
    trading_days: int = 252
    risk_free_rate: float = 0.03
    price_interval: str = "1d"
    auto_adjust: bool = True
    benchmark_ticker: str = "SPY"
    return_method: str = "pct"
    drawdown_method: str = "full"
    drawdown_window_days: int = 365
    beta_lookback_days: int = 365
    risk_free_source: str = "constant"
    risk_free_ticker: str = "^IRX"
    model: str | None = "gpt-4o-mini"


@app.post("/api/portfolio")
async def analyze_portfolio(request: PortfolioRequest):
    if not request.holdings:
        raise HTTPException(status_code=400, detail="Holdings are required.")

    holdings = [
        PortfolioInput(ticker=h.ticker.upper().strip(), weight=h.weight)
        for h in request.holdings
        if h.ticker
    ]
    holdings = _normalize_weights(holdings)

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=request.lookback_days)

    cache_key = _hash_payload(
        {
            "holdings": [h.model_dump() for h in holdings],
            "start": str(start_date),
            "end": str(end_date),
            "trading_days": request.trading_days,
            "risk_free_rate": request.risk_free_rate,
            "price_interval": request.price_interval,
            "auto_adjust": request.auto_adjust,
            "benchmark_ticker": request.benchmark_ticker,
            "return_method": request.return_method,
            "drawdown_method": request.drawdown_method,
            "drawdown_window_days": request.drawdown_window_days,
            "beta_lookback_days": request.beta_lookback_days,
            "risk_free_source": request.risk_free_source,
            "risk_free_ticker": request.risk_free_ticker,
        }
    )
    cached = _cache_get(PORTFOLIO_CACHE, cache_key)
    if cached is not None:
        logger.debug("portfolio cache hit for %s", cache_key)
        return cached

    portfolio_data = _portfolio_stats(
        holdings,
        start_date,
        end_date,
        request.trading_days,
        request.risk_free_rate,
        request.price_interval,
        request.auto_adjust,
        request.benchmark_ticker,
        request.return_method,
        request.drawdown_method,
        request.drawdown_window_days,
        request.beta_lookback_days,
        request.risk_free_source,
        request.risk_free_ticker,
    )
    if portfolio_data.get("error"):
        raise HTTPException(status_code=404, detail=portfolio_data["error"])

    payload = {
        "holdings": portfolio_data["holdings"],
        "sector_allocation": portfolio_data["sector_allocation"],
        "stats": portfolio_data["stats"],
    }
    ai_summary = await _openai_portfolio_brief(payload, model=request.model)

    suggested_stats = None
    suggested_best = None
    suggested_candidates = []
    suggestions = (
        ai_summary.get("suggested_portfolios") if isinstance(ai_summary, dict) else None
    )
    if suggestions and isinstance(suggestions, list):
        for idx, suggestion in enumerate(suggestions):
            try:
                inputs = [
                    (
                        PortfolioInput(ticker=item["ticker"], weight=item["weight"])
                        if isinstance(item, dict)
                        else item
                    )
                    for item in suggestion
                ]
                inputs = _normalize_weights(inputs)
                stats = _portfolio_stats(
                    inputs,
                    start_date,
                    end_date,
                    request.trading_days,
                    request.risk_free_rate,
                    request.price_interval,
                    request.auto_adjust,
                    request.benchmark_ticker,
                    request.return_method,
                    request.drawdown_method,
                    request.drawdown_window_days,
                    request.beta_lookback_days,
                    request.risk_free_source,
                    request.risk_free_ticker,
                )
                if not stats.get("error"):
                    suggested_candidates.append({"inputs": inputs, "stats": stats})
            except Exception:
                logger.exception("failed to compute suggested portfolio stats %s", idx)

    if suggested_candidates:
        suggested_candidates.sort(
            key=lambda x: (
                x["stats"]["stats"].get("sharpe_ratio") or -999,
                x["stats"]["stats"].get("annualized_return") or -999,
            ),
            reverse=True,
        )
        best = suggested_candidates[0]
        suggested_best = best["inputs"]
        suggested_stats = best["stats"]

    response = {
        "as_of": end_date.isoformat(),
        "holdings": portfolio_data["holdings"],
        "sector_allocation": portfolio_data["sector_allocation"],
        "stats": portfolio_data["stats"],
        "suggested_portfolio": suggested_stats,
        "suggested_holdings": (
            [h.model_dump() for h in suggested_best] if suggested_best else None
        ),
        "ai_summary": ai_summary,
        "disclaimer": "This is for educational purposes only and not investment advice.",
        "assumptions": {
            "lookback_days": request.lookback_days,
            "trading_days": request.trading_days,
            "risk_free_rate": request.risk_free_rate,
            "price_interval": request.price_interval,
            "auto_adjust": request.auto_adjust,
            "benchmark_ticker": request.benchmark_ticker,
            "return_method": request.return_method,
            "drawdown_method": request.drawdown_method,
            "drawdown_window_days": request.drawdown_window_days,
            "beta_lookback_days": request.beta_lookback_days,
            "risk_free_source": request.risk_free_source,
            "risk_free_ticker": request.risk_free_ticker,
        },
    }
    if isinstance(ai_summary, dict) and suggested_stats:
        ai_summary["comparison"] = _compare_portfolios(portfolio_data, suggested_stats)
    _cache_set(PORTFOLIO_CACHE, cache_key, response)
    return response


class ComparisonRequest(BaseModel):
    portfolio_a: list[PortfolioInput]
    portfolio_b: list[PortfolioInput]
    lookback_days: int = 365
    trading_days: int = 252
    risk_free_rate: float = 0.03
    price_interval: str = "1d"
    auto_adjust: bool = True
    model: str | None = "gpt-4o-mini"


@app.post("/api/compare-portfolios")
async def compare_portfolios(request: ComparisonRequest):
    if not request.portfolio_a or not request.portfolio_b:
        raise HTTPException(status_code=400, detail="Both portfolios are required.")

    def _prep_holdings(h_list):
        h = [
            PortfolioInput(ticker=x.ticker.upper().strip(), weight=x.weight)
            for x in h_list
            if x.ticker
        ]
        return _normalize_weights(h)

    ha = _prep_holdings(request.portfolio_a)
    hb = _prep_holdings(request.portfolio_b)

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=request.lookback_days)

    def _get_stats(h):
        return _portfolio_stats(
            h,
            start_date,
            end_date,
            request.trading_days,
            request.risk_free_rate,
            request.price_interval,
            request.auto_adjust,
            "SPY",
            "pct",
            "full",
            365,
            365,
            "constant",
            "^IRX",
        )

    stats_a = _get_stats(ha)
    stats_b = _get_stats(hb)

    if stats_a.get("error") or stats_b.get("error"):
        raise HTTPException(
            status_code=404,
            detail=f"Error fetching data: {stats_a.get('error') or stats_b.get('error')}",
        )

    payload = {
        "portfolio_a": {
            "stats": stats_a["stats"],
            "sector_allocation": stats_a["sector_allocation"],
            "holdings": stats_a["holdings"],
        },
        "portfolio_b": {
            "stats": stats_b["stats"],
            "sector_allocation": stats_b["sector_allocation"],
            "holdings": stats_b["holdings"],
        },
    }

    ai_comparison = await _openai_comparison_brief(payload, model=request.model)

    return {
        "as_of": end_date.isoformat(),
        "portfolio_a": stats_a,
        "portfolio_b": stats_b,
        "ai_comparison": ai_comparison,
        "disclaimer": "This is for educational purposes only and not investment advice.",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
