import os
import yfinance as yf
import openai
from typing import List, Dict, Optional
import json

class QuantAgent:
    def __init__(self, ticker_symbol: str):
        self.ticker = ticker_symbol.upper()
        self.stock = yf.Ticker(self.ticker)
    
    def get_snapshot(self) -> Dict:
        info = self.stock.info
        return {
            "price": info.get('currentPrice') or info.get('regularMarketPrice'),
            "currency": info.get('currency', 'USD'),
            "pe_ratio": info.get('trailingPE'),
            "debt_to_equity": info.get('debtToEquity'),
            "free_cash_flow": info.get('freeCashflow'),
            "market_cap": info.get('marketCap'),
            "analyst_ratings": {
                "recommendation": info.get('recommendationKey'),
                "target_mean": info.get('targetMeanPrice'),
                "number_of_analysts": info.get('numberOfAnalystOpinions')
            },
            "earnings": self.stock.earnings.to_dict() if hasattr(self.stock, 'earnings') else {}
        }

class NewsAgent:
    def __init__(self, ticker_symbol: str):
        self.ticker = ticker_symbol.upper()
    
    async def get_recent_headlines(self) -> List[Dict]:
        # In a real app, use Tavily/Serper. Mocking news aggregation for now.
        return [
            {"title": f"{self.ticker} reports better than expected Q4 results", "source": "Finance Daily", "sentiment": "positive"},
            {"title": f"New regulatory scrutiny on {self.ticker}'s cloud division", "source": "Tech Wire", "sentiment": "negative"},
            {"title": f"Analysts raise price targets for {self.ticker} ahead of keynote", "source": "Market Watch", "sentiment": "positive"}
        ]

class Orchestrator:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    async def synthesize_report(self, ticker: str, quant_data: Dict, news_data: List[Dict]) -> Dict:
        prompt = f"""
        You are a Lead Financial Analyst. Synthesize a report for {ticker} based on raw data.
        
        Quantitative Data: {json.dumps(quant_data)}
        Recent News: {json.dumps(news_data)}
        
        Provide the following sections in JSON format:
        1. investment_thesis: A 3-sentence summary.
        2. risk_reward: Breakdown of upside vs downside.
        3. portfolio_fit: Guidance for different risk profiles.
        4. bottom_line: A one-sentence recommendation.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
