"""
FinSight AI — Risk Agent
Uses the google-genai SDK with manual function calling to compute risk metrics
and produce a structured risk assessment.
"""

import json
import os
import time

from google import genai
from google.genai import types

from agents.mcp_tools import GEMINI_TOOL_DECLARATIONS, execute_tool


client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

_TOOLS = types.Tool(function_declarations=GEMINI_TOOL_DECLARATIONS)
_CONFIG = types.GenerateContentConfig(
    tools=[_TOOLS],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    system_instruction=(
        "You are the Risk Agent in a multi-agent financial analysis system.\n"
        "Your job:\n"
        "1. Calculate risk metrics using calculate_risk_metrics\n"
        "2. Optionally fetch market data using get_market_data for price context\n"
        "3. Produce a structured risk assessment\n\n"
        "Include: risk tier (Conservative/Moderate/Aggressive/Speculative), key risk "
        "factors (volatility, drawdown, beta), comparison to SPY, position-sizing "
        "guidance based on the risk score, and specific downside scenarios with "
        "estimated loss ranges. Be quantitative — use the actual numbers from your "
        "tool calls. 200-400 words."
    ),
)


def run_risk_agent(state: dict) -> dict:
    start = time.time()
    symbol = state["symbol"]
    query = state["query"]

    research_context = ""
    if state.get("research_summary"):
        research_context = (
            f"\n\nThe Research Agent already gathered this context:\n"
            f"{state['research_summary'][:1000]}"
        )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=(
                f"Assess the risk profile of {symbol}. User query: {query}\n\n"
                f"Use calculate_risk_metrics to get quantitative risk data, then write "
                f"your risk assessment.{research_context}"
            ))],
        )
    ]

    tools_called = []
    risk_metrics = None
    final_text = ""

    max_iterations = 8
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
                if fc.name == "calculate_risk_metrics":
                    risk_metrics = parsed
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
        "risk_metrics": risk_metrics,
        "risk_assessment": final_text,
        "tools_called": tools_called,
        "processing_time_seconds": round(
            (state.get("processing_time_seconds") or 0) + elapsed, 2
        ),
    }
