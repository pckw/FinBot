from langchain.vectorstores import Chroma

class chromaDB():
    def __init__(self) -> None:
        pass

    def create_vectordb(documents, embedding, persist_directory='./data'):
        return Chroma.from_documents(
            documents,
            embedding=embedding,
            persist_directory=persist_directory
            )

    def read_vectordb(embedding, persist_directory='./data'):
        return Chroma(embedding_function=embedding, persist_directory=persist_directory)