from langchain.vectorstores import Chroma


class chromaDB():
    def __init__(self) -> None:
        """
        Initializes the object.

        Parameters:
        - None

        Returns:
        - None
        """
        pass

    def create_vectordb(documents, embedding: object, persist_directory: str) -> object:
        """
        Creates a new VectorDB instance and adds the given documents to it.

        Parameters:
            documents (list): A list of documents to be added to the VectorDB.
            embedding (function): The embedding function to be used for generating document embeddings.
            persist_directory (str): The directory where the VectorDB should persist its data.

        Returns:
            vectordb (VectorDB): The newly created VectorDB instance.
        """
        vectordb = Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory
        )
        ids = vectordb.get()['ids']
        if ids:
            vectordb._collection.delete(ids=vectordb.get()['ids'])
        vectordb.add_documents(documents=documents)
        vectordb.persist()
        return vectordb

    def read_vectordb(
            embedding: object,
            persist_directory: str = './data'
    ) -> object:
        """
        Read from a Chroma object from a given embedding function and persist directory.

        Args:
            embedding (function): The embedding function to be used by the Chroma object.
            persist_directory (str, optional): The directory to persist the Chroma object. Defaults to './data'.

        Returns:
            Chroma: The Chroma object generated from the embedding function and persist directory.
        """
        return Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory
        )
