from fastapi import APIRouter, HTTPException, Request
from app.models.query import QueryRequest, QueryResponse
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.base.response.schema import Response
router = APIRouter()

@router.post("/query", 
    response_model=QueryResponse,
    summary="Query the RAG system",
    description="Send a query to the RAG system and receive an answer with sources"
)
async def query_rag(
    query_request: QueryRequest,
    request: Request,
) -> QueryResponse:
    """
    Query the RAG system with the following parameters:

    Args:
        query_request (QueryRequest): A query request representing the query to be made
        request (Request): A FastAPI request object where we have the current app state

    Returns:
        QueryResponse: A query response representing the answer and sources where the sources are the nodes in the vector store
    """
    index: VectorStoreIndex = request.app.state.index
    query_engine: BaseQueryEngine = index.as_query_engine(llm=request.app.state.mini_language_model)
    try:
        response: Response = query_engine.query(query_request.query)
        sources = [
            {
                "id": source.node.id_,
                "metadata": source.node.metadata,
                "score": source.score,
                "text": source.node.text
            }
            for source in response.source_nodes
        ]
        return QueryResponse(
            answer=response.response,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"} 