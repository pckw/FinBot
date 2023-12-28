from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient


class qdrantDB():
    def __init__(self) -> None:
        pass

    def create_vectordb(documents, embedding, persist_directory):
        vectordb = Qdrant.from_documents(
            documents=documents,
            embedding=embedding,
            path=persist_directory,
            collection_name="collection",
            force_recreate=True,
        )
        return vectordb

    def read_vectordb(embedding, persist_directory='./data'):
        client = QdrantClient(path=persist_directory)
        return Qdrant(
            client=client,
            embeddings=embedding,
            collection_name="collection",
        )
