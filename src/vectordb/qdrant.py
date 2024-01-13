from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient


class qdrantDB():
    def __init__(self) -> None:
        """
        Initializes an instance of the class.

        Parameters:
            None

        Returns:
            None
        """
        pass

    def create_vectordb(
            documents,
            embedding: object,
            persist_directory: str
    ) -> object:
        """
        Create a VectorDB instance with the given documents, embedding, and persist directory.

        Args:
            documents (list[str]): A list of documents to be indexed.
            embedding (str): The type of embedding used for indexing the documents.
            persist_directory (str): The directory where the VectorDB index will be persisted.

        Returns:
            VectorDB: The created VectorDB instance.

        Raises:
            None
        """
        vectordb = Qdrant.from_documents(
            documents=documents,
            embedding=embedding,
            path=persist_directory,
            collection_name="collection",
            force_recreate=True,
        )
        return vectordb

    def read_vectordb(embedding, persist_directory='./data') -> object:
        """
        Initializes and returns a Qdrant instance with the specified embedding and persist directory.

        Parameters:
            embedding (function): The embedding to be used for the Qdrant instance.
            persist_directory (str): The directory where the Qdrant instance will persist its data. Defaults to './data'.

        Returns:
            Qdrant: The initialized Qdrant instance.

        """
        client = QdrantClient(path=persist_directory)
        return Qdrant(
            client=client,
            embeddings=embedding,
            collection_name="collection",
        )
