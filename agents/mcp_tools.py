"""
FinSight AI — MCP Tools Bridge
Defines tool schemas for Claude API tool-use and executes them locally.

This module serves two purposes:
1. Provides tool definitions (JSON schemas) that get passed to Claude's `tools` param
2. Executes tool calls by delegating to the same functions exposed by the MCP server

In production, these would be called via MCP client → MCP server.
This bridge lets the LangGraph agents work both with and without the MCP server running.
"""

import json
import math
from datetime import datetime

import yfinance as yf


# ── Tool Definitions (Claude API format) ──────────────────────────────────────

FINANCIAL_TOOLS = [
    {
        "name": "get_market_data",
        "description": (
            "Fetch real-time market data for a stock symbol including price, "
            "PE ratio, market cap, sector, 52-week range, and recent price history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker (e.g., AAPL, NVDA, TSLA)",
                },
                "period": {
                    "type": "string",
                    "description": "Data period: 1mo, 3mo, 6mo, 1y, 2y, 5y",
                    "default": "6mo",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "calculate_risk_metrics",
        "description": (
            "Compute risk metrics for a stock: volatility, Sharpe ratio, "
            "max drawdown, beta vs SPY, and a composite risk score (0-100)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker",
                },
                "period": {
                    "type": "string",
                    "description": "Historical period: 6mo, 1y, 2y",
                    "default": "1y",
                },
                "risk_free_rate": {
                    "type": "number",
                    "description": "Annual risk-free rate for Sharpe calculation",
                    "default": 0.05,
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "compare_peers",
        "description": (
            "Compare a stock against sector peers on PE, market cap, beta, "
            "margins, and revenue growth. Auto-detects peers if not specified."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Primary stock ticker",
                },
                "peers": {
                    "type": "string",
                    "description": "Comma-separated peer tickers (e.g., 'MSFT,GOOG,META'). Auto-detects if empty.",
                    "default": "",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_financial_summary",
        "description": (
            "Get financial summary: revenue, net income, margins, debt levels, "
            "cash flow, and valuation ratios for the most recent fiscal year."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker",
                },
            },
            "required": ["symbol"],
        },
    },
]


# ── Gemini-format declarations (derived from FINANCIAL_TOOLS) ─────────────────
# Gemini's function_declarations use "parameters" where Claude uses "input_schema".
# We derive them from the same source so there's one place to edit tool specs.

GEMINI_TOOL_DECLARATIONS = [
    {
        "name": t["name"],
        "description": t["description"],
        "parameters": t["input_schema"],
    }
    for t in FINANCIAL_TOOLS
]


# ── Tool Execution ────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Execute a tool call and return the result as a JSON string.
    These are the same implementations as the MCP server tools.
    """
    dispatch = {
        "get_market_data": _get_market_data,
        "calculate_risk_metrics": _calculate_risk_metrics,
        "compare_peers": _compare_peers,
        "get_financial_summary": _get_financial_summary,
    }

    handler = dispatch.get(tool_name)
    if not handler:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        return handler(**tool_input)
    except Exception as e:
        return json.dumps({"error": f"{tool_name} failed: {str(e)}"})


def _get_market_data(symbol: str, period: str = "6mo") -> str:
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        hist = ticker.history(period=period)

        if hist.empty:
            return json.dumps({"error": f"No data found for {symbol}"})

        recent = hist.tail(5)
        price_history = [
            {
                "date": idx.strftime("%Y-%m-%d"),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"]),
            }
            for idx, row in recent.iterrows()
        ]

        return json.dumps({
            "symbol": symbol.upper(),
            "current_price": round(info.get("currentPrice", info.get("regularMarketPrice", 0)), 2),
            "previous_close": round(info.get("previousClose", 0), 2),
            "pe_ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else None,
            "forward_pe": round(info.get("forwardPE", 0), 2) if info.get("forwardPE") else None,
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "fifty_two_week_high": round(info.get("fiftyTwoWeekHigh", 0), 2),
            "fifty_two_week_low": round(info.get("fiftyTwoWeekLow", 0), 2),
            "avg_volume": info.get("averageVolume"),
            "dividend_yield": round(info.get("dividendYield", 0), 2) if info.get("dividendYield") else None,
            "beta": round(info.get("beta", 0), 3) if info.get("beta") else None,
            "recent_prices": price_history,
            "data_period": period,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "symbol": symbol})


def _calculate_risk_metrics(symbol: str, period: str = "1y", risk_free_rate: float = 0.05) -> str:
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period=period)

        if len(hist) < 30:
            return json.dumps({"error": f"Insufficient data for {symbol}"})

        closes = hist["Close"]
        daily_returns = closes.pct_change().dropna()

        volatility = float(daily_returns.std() * math.sqrt(252))
        total_return = float((closes.iloc[-1] / closes.iloc[0]) - 1)
        trading_days = len(closes)
        annualized_return = float((1 + total_return) ** (252 / trading_days) - 1)
        sharpe = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

        rolling_max = closes.cummax()
        drawdown = (closes - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        # Beta vs SPY
        spy = yf.Ticker("SPY").history(period=period)
        if len(spy) >= 30:
            spy_returns = spy["Close"].pct_change().dropna()
            min_len = min(len(daily_returns), len(spy_returns))
            stock_r = daily_returns.iloc[-min_len:]
            market_r = spy_returns.iloc[-min_len:]
            covariance = float(stock_r.cov(market_r))
            market_var = float(market_r.var())
            beta = covariance / market_var if market_var > 0 else 1.0
        else:
            beta = 1.0

        vol_score = min(volatility / 0.8, 1.0) * 30
        dd_score = min(abs(max_drawdown) / 0.5, 1.0) * 25
        beta_score = min(abs(beta) / 2.0, 1.0) * 25
        sharpe_penalty = max(0, (1 - sharpe) / 2) * 20
        risk_score = round(vol_score + dd_score + beta_score + sharpe_penalty, 1)

        if risk_score < 25:
            risk_tier = "Conservative"
        elif risk_score < 50:
            risk_tier = "Moderate"
        elif risk_score < 75:
            risk_tier = "Aggressive"
        else:
            risk_tier = "Speculative"

        return json.dumps({
            "symbol": symbol.upper(),
            "volatility": round(volatility, 4),
            "annualized_return": round(annualized_return, 4),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": round(max_drawdown, 4),
            "beta": round(beta, 3),
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "calculation_params": {
                "period": period,
                "risk_free_rate": risk_free_rate,
                "trading_days_used": trading_days,
            },
        })
    except Exception as e:
        return json.dumps({"error": str(e), "symbol": symbol})


def _compare_peers(symbol: str, peers: str = "") -> str:
    try:
        primary = yf.Ticker(symbol.upper())
        primary_info = primary.info

        if peers:
            peer_list = [p.strip().upper() for p in peers.split(",")]
        else:
            sector = primary_info.get("sector", "")
            sector_peers = {
                "Technology": ["AAPL", "MSFT", "GOOG", "META", "NVDA", "AMZN"],
                "Financial Services": ["JPM", "BAC", "GS", "MS", "WFC", "C"],
                "Healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABT", "LLY"],
                "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX"],
                "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX"],
            }
            candidates = sector_peers.get(sector, [])
            peer_list = [p for p in candidates if p != symbol.upper()][:4]

        comparisons = []
        for ticker_symbol in [symbol.upper()] + peer_list:
            try:
                info = yf.Ticker(ticker_symbol).info
                comparisons.append({
                    "symbol": ticker_symbol,
                    "price": round(info.get("currentPrice", 0), 2),
                    "pe_ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else None,
                    "market_cap_B": round(info.get("marketCap", 0) / 1e9, 1) if info.get("marketCap") else None,
                    "beta": round(info.get("beta", 0), 3) if info.get("beta") else None,
                    "profit_margin": round(info.get("profitMargins", 0) * 100, 1) if info.get("profitMargins") else None,
                    "revenue_growth": round(info.get("revenueGrowth", 0) * 100, 1) if info.get("revenueGrowth") else None,
                })
            except Exception:
                continue

        return json.dumps({
            "primary": symbol.upper(),
            "sector": primary_info.get("sector", "N/A"),
            "comparisons": comparisons,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def _get_financial_summary(symbol: str) -> str:
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        return json.dumps({
            "symbol": symbol.upper(),
            "company_name": info.get("longName", symbol),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "employees": info.get("fullTimeEmployees"),
            "financials": {
                "revenue": info.get("totalRevenue"),
                "net_income": info.get("netIncomeToCommon"),
                "profit_margin": round(info.get("profitMargins", 0) * 100, 2) if info.get("profitMargins") else None,
                "operating_margin": round(info.get("operatingMargins", 0) * 100, 2) if info.get("operatingMargins") else None,
                "revenue_growth_yoy": round(info.get("revenueGrowth", 0) * 100, 1) if info.get("revenueGrowth") else None,
            },
            "balance_sheet": {
                "total_cash": info.get("totalCash"),
                "total_debt": info.get("totalDebt"),
                "debt_to_equity": round(info.get("debtToEquity", 0), 2) if info.get("debtToEquity") else None,
                "current_ratio": round(info.get("currentRatio", 0), 2) if info.get("currentRatio") else None,
            },
            "valuation": {
                "pe_ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else None,
                "forward_pe": round(info.get("forwardPE", 0), 2) if info.get("forwardPE") else None,
                "price_to_book": round(info.get("priceToBook", 0), 2) if info.get("priceToBook") else None,
            },
        })
    except Exception as e:
        return json.dumps({"error": str(e), "symbol": symbol})
