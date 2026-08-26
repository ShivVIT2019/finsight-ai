"""
FinSight AI — Research Agent
Uses the google-genai SDK with manual function calling to fetch market data,
financial summaries, and peer comparisons via the shared financial tools.

The loop is explicit (not automatic function calling) so the agentic tool-use
pattern is visible: model proposes a function_call -> we execute the real
yfinance tool -> we send the result back as a function_response -> repeat until
the model stops requesting tools.
"""

import json
import os
import time
from datetime import date

from google import genai
from google.genai import types

from agents.mcp_tools import GEMINI_TOOL_DECLARATIONS, execute_tool


client = genai.Client(vertexai=True, project="project-13b1a293-3e33-4184-8d9", location="us-central1")
MODEL = "gemini-2.5-flash"

_TOOLS = types.Tool(function_declarations=GEMINI_TOOL_DECLARATIONS)
_CONFIG = types.GenerateContentConfig(
    tools=[_TOOLS],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    system_instruction=(
        f"You are the Research Agent in a multi-agent financial analysis system.\n"
        f"Today's date is {date.today().isoformat()}. Use this exact date whenever "
        f"you reference the analysis date; do not invent a date.\n"
        "Your job:\n"
        "1. Fetch real-time market data using get_market_data\n"
        "2. Pull financial summary (revenue, margins, cash flow) using get_financial_summary\n"
        "3. Compare against sector peers using compare_peers\n"
        "4. Synthesize everything into a structured research brief\n\n"
        "Call all three tools before writing your analysis. Use specific numbers "
        "from the tool results. Flag any missing data. Keep the brief data-rich, "
        "300-500 words."
    ),
)


def run_research_agent(state: dict) -> dict:
    start = time.time()
    symbol = state["symbol"]
    query = state["query"]

    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=(
                f"Analyze {symbol}. User query: {query}\n\n"
                f"Use the available tools to gather market data, a financial summary, "
                f"and a peer comparison for {symbol}, then write your research brief."
            ))],
        )
    ]

    tools_called = []
    market_data = None
    financial_summary = None
    peer_comparison = None
    final_text = ""

    max_iterations = 10
    for _ in range(max_iterations):
        response = client.models.generate_content(
            model=MODEL, contents=contents, config=_CONFIG
        )

        candidate = response.candidates[0]
        contents.append(candidate.content)

        fcalls = response.function_calls or []
        if not fcalls:
            final_text = response.text or ""
            break

        response_parts = []
        for fc in fcalls:
            args = dict(fc.args) if fc.args else {}
            tools_called.append({"tool": fc.name, "input": args})

            result = execute_tool(fc.name, args)
            try:
                parsed = json.loads(result)
                if fc.name == "get_market_data":
                    market_data = parsed
                elif fc.name == "get_financial_summary":
                    financial_summary = parsed
                elif fc.name == "compare_peers":
                    peer_comparison = parsed
            except (json.JSONDecodeError, TypeError):
                parsed = {"raw": result}

            response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": parsed},
                )
            )

        contents.append(types.Content(role="user", parts=response_parts))

    elapsed = time.time() - start

    return {
        "research_summary": final_text,
        "market_data": market_data,
        "financial_summary": financial_summary,
        "peer_comparison": peer_comparison,
        "tools_called": tools_called,
        "processing_time_seconds": round(elapsed, 2),
        "model_used": MODEL,
    }
