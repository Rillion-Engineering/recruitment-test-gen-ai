from pathlib import Path
from qdrant_client import QdrantClient
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.core.readers import SimpleDirectoryReader
import os
from dotenv import load_dotenv

load_dotenv()


def put_documents_in_vector_store(
    documents_dir: Path,
    qdrant_collection_name: str,
    chunk_size: int,
    overlap: int,
):
    """Create and return a VectorStoreIndex from the given documents directory and Qdrant client."""
    qdrant_url = os.getenv("QDRANT_URL","http://127.0.0.1")
    qdrant_port = os.getenv("QDRANT_PORT",6333)
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(url=qdrant_url, port=qdrant_port, api_key=qdrant_api_key)
    # Create vector store
    vector_store = QdrantVectorStore(
        client=client, collection_name=qdrant_collection_name
    )
    print("Vector store created successfully")

    embedding_model = AzureOpenAIEmbedding(
        model="text-embedding-3-large",
        deployment_name="text-embedding-3-large",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2025-03-01-preview",
    )
    # Load documents
    reader = SimpleDirectoryReader(input_dir=documents_dir)
    documents = reader.load_data(show_progress=True, num_workers=1)

    print(f"Loaded {len(documents)} documents from {documents_dir}")
    docstore = SimpleDocumentStore()
    # Create ingestion pipeline
    pipeline = IngestionPipeline(
        name=f"{documents_dir}_pipeline",
        project_name=f"{qdrant_url}_{qdrant_collection_name}_project",
        transformations=[
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap),
            embedding_model,
        ],
        docstore=docstore,
        vector_store=vector_store,
    )
    pipeline.run(documents=documents, num_workers=1, show_progress=True)


if __name__ == "__main__":
    put_documents_in_vector_store(
        documents_dir=Path("data"),
        qdrant_collection_name="data",
        chunk_size=512,
        overlap=20,
    )
