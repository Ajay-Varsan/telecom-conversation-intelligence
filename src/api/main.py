import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes_analyze import router as analyze_router, corpus_loader, analyze_batch_conversation
from src.api.routes_qa import router as qa_router
from src.api.routes_health import router as health_router

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler: Pre-seed sample conversations from dataset to populate rollups."""
    try:
        samples = corpus_loader.load_sample_conversations(limit_convs=15)
        for s in samples[:8]:
            await analyze_batch_conversation(s)
        print(f"[STARTUP] Successfully analyzed and indexed {min(8, len(samples))} initial conversations.")
    except Exception as e:
        print(f"[STARTUP WARNING] Failed to pre-seed corpus: {e}")
    yield

app = FastAPI(
    title="Telecom Contact Center Conversation Analytics Microservice",
    description="Automated 100% conversation intelligence, live assist next-best-actions, grounded QA evaluation, and supervisor rollups.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend clients / dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(analyze_router)
app.include_router(qa_router)
app.include_router(health_router)

from fastapi.responses import RedirectResponse


@app.get("/")
@app.get("/dashboard")
async def redirect_to_streamlit():
    """Redirects web browser requests directly to the primary Streamlit Dashboard."""
    return RedirectResponse(url="http://localhost:8501", status_code=307)


@app.get("/api")
async def api_info():
    """Headless REST API microservice gateway metadata."""
    return {
        "service": "Telecom Conversation Analytics Microservice",
        "status": "Active",
        "docs_url": "/docs",
        "health_url": "/health",
        "primary_dashboard": "http://localhost:8501"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
