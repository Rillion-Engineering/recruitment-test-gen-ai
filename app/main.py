from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from app.core.config import settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from app.api.endpoints import query
from llama_index.core import Settings as LlamaIndexSettings
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    gpt4o_mini_language_model = AzureOpenAI(
        model="gpt-4o-mini",
        deployment_name="gpt-4o-mini",
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY.get_secret_value(),
        api_version="2025-03-01-preview",
        timeout=15,
        additional_kwargs={"stream_options": {"include_usage": True}},
    )
    embedding_model = AzureOpenAIEmbedding(
        model="text-embedding-3-large",
        deployment_name="text-embedding-3-large",
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY.get_secret_value(),
        api_version="2025-03-01-preview",
    )

    # Configure LlamaIndex settings
    LlamaIndexSettings.llm = gpt4o_mini_language_model
    LlamaIndexSettings.embed_model = embedding_model

    # Set up sync Qdrant client and vector store
    qdrant_client = QdrantClient(
        url=settings.QDRANT_URL if settings.QDRANT_URL else None,
        port=int(settings.QDRANT_PORT) if settings.QDRANT_PORT is not None else None,
        timeout=30,
    )
    vector_store = QdrantVectorStore(
        client=qdrant_client, collection_name=settings.QDRANT_COLLECTION_NAME
    )

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    print("Vector store and index initialized successfully.")

    # Store these in the app state
    app.state.index = index
    app.state.qdrant_client = qdrant_client
    app.state.embedding_model = embedding_model
    app.state.mini_language_model = gpt4o_mini_language_model

    yield
    # ... cleanup code ...
    # The code after yield runs during application shutdown
    qdrant_client.close()


def create_app():
    print("\n\n")
    print("███╗   ███╗██╗███╗   ██╗██████╗  ██████╗██████╗  █████╗ ███████╗████████╗")
    print("████╗ ████║██║████╗  ██║██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝")
    print("██╔████╔██║██║██╔██╗ ██║██║  ██║██║     ██████╔╝███████║█████╗     ██║   ")
    print("██║╚██╔╝██║██║██║╚██╗██║██║  ██║██║     ██╔══██╗██╔══██║██╔══╝     ██║   ")
    print("██║ ╚═╝ ██║██║██║ ╚████║██████╔╝╚██████╗██║  ██║██║  ██║██║        ██║   ")
    print("╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝  ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝        ╚═╝   ")
    print("████████╗███████╗███████╗████████╗  █████╗ ██████╗ ██╗")
    print("╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝ ██╔══██╗██╔══██╗██║")
    print("   ██║   █████╗  ███████╗   ██║    ███████║██████╔╝██║")
    print("   ██║   ██╔══╝  ╚════██║   ██║    ██╔══██║██╔═══╝ ██║")
    print("   ██║   ███████╗███████║   ██║    ██║  ██║██║     ██║")
    print("   ╚═╝   ╚══════╝╚══════╝   ╚═╝    ╚═╝  ╚═╝╚═╝     ╚═╝ v%s" % "0.1.0")
    print("\n")
    print("| 📚 Documentation 📚")
    print("| redoc: http://0.0.0.0:8003/redoc")
    print("| swagger: http://0.0.0.0:8003/docs")
    print("\n\n")

    app = FastAPI(
        title="Mindcraft Test API",
        description="This is a test API for recruitment purposes",
        lifespan=lifespan,
        version="0.1.0",
    )
    app.include_router(query.router, prefix=settings.API_V1_STR)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003, reload=True)
