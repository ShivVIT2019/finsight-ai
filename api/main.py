"""
FinSight AI — FastAPI Backend
HTTP entry point for the multi-agent analysis pipeline.
"""

from fastapi import FastAPI, Request

from graph.pipeline import run_analysis

app = FastAPI(
    title="FinSight AI",
    description="Real-time financial intelligence via a multi-agent LangGraph pipeline",
)


@app.get("/")
def root():
    return {"message": "FinSight AI is running. See /docs for the API."}


@app.post("/analyze")
async def analyze(request: Request):
    """
    Analyze a stock and return an investment brief.

    Body: {"query": "Should I invest in NVDA?", "symbol": "NVDA"}
    symbol is optional — extracted from the query if omitted.
    """
    payload = await request.json()
    query = payload.get("query", "")
    symbol = payload.get("symbol", "")
    if not query:
        return {"error": "Missing required field: 'query'"}
    return run_analysis(query=query, symbol=symbol)


@app.get("/healthz")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
