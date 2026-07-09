"""Reference Agentic RAG application — entry point."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from .api.routes import router as search_router


def create_app() -> FastAPI:
    """Application factory."""
    application = FastAPI(
        title="Citadel RAG Factory — Reference Agentic RAG",
        description="Production-grade RAG with adaptive retrieval, multi-hop reasoning, and self-correction.",
        version="0.1.0",
    )

    application.include_router(search_router)

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "reference-agentic-rag"}

    @application.get("/")
    async def root() -> dict:
        return {
            "service": "reference-agentic-rag",
            "docs": "/docs",
            "patterns": ["naive", "hybrid", "graph", "multimodal", "agentic"],
            "endpoints": {
                "search": "/api/v1/search",
                "ingest": "/api/v1/ingest",
                "strategies": "/api/v1/strategies",
            },
        }

    return application


app = create_app()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
