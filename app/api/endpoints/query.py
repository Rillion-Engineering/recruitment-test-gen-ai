from typing import cast
from fastapi import APIRouter, Request
from openai import AsyncAzureOpenAI
from qdrant_client import AsyncQdrantClient
from app.models.query import QueryRequest, QueryResponse
from app.core.config import settings
import json

router = APIRouter()


def _convert_nodes_to_sources(node_results):
    sources = []
    for point in node_results.points:
        payload = point.payload
        if not payload:
            continue

        node_content = json.loads(payload.get("_node_content", "{}"))

        if not node_content:
            continue
        exclude_llm_metadata = node_content.get("excluded_llm_metadata_keys", [])
        text_template = node_content.get(
            "text_template", "Metadata: {metadata_str}\n-----\nContent: {content}"
        )
        metadata_template = node_content.get("metadata_template", "{key}: {value}")
        metadata_seperator = node_content.get("metadata_seperator", "\n")

        metadata_str = metadata_seperator.join(
            [
                metadata_template.format(key=k, value=v)
                for k, v in node_content.get("metadata", {}).items()
                if k not in exclude_llm_metadata
            ]
        )
        content = text_template.format(
            metadata_str=metadata_str, content=node_content.get("text", "")
        )
        sources.append(
            {
                "id": point.id,
                "content": content,
                "score": point.score,
            }
        )
    return sources


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG system",
    description="Send a query to the RAG system and receive an answer with sources",
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
    embedding_client = cast(AsyncAzureOpenAI, request.app.state.openai_async_embedder)
    async_open_ai_client = cast(AsyncAzureOpenAI, request.app.state.openai_async_client)
    async_qdrant_client = cast(AsyncQdrantClient, request.app.state.qdrant_client)

    embedding_result = await embedding_client.embeddings.create(
        input=query_request.query, model="text-embedding-3-large"
    )
    embedding = embedding_result.data[0].embedding
    print("retrieved embedding")
    node_results = await async_qdrant_client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=embedding,
        limit=3,
    )
    sources = _convert_nodes_to_sources(node_results)
    print(f"retrieved {len(sources)} nodes")
    system_prompt = """
You are a helpful assistant. 
You have access to a the following sources of information based on the users question.

SOURCES:

"""
    system_prompt += "\n\n".join([source["content"] for source in sources])

    openai_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query_request.query},
    ]
    response = await async_open_ai_client.chat.completions.create(
        model="gpt-4o",
        messages=openai_messages,
        max_completion_tokens=512,
    )
    response = response.choices[0].message.content
    return QueryResponse(answer=response, sources=sources)


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
