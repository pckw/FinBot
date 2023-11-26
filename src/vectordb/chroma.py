from langchain.vectorstores import Chroma

class chromaDB():
    def __init__(self) -> None:
        pass

    def vectordb(documents, embedding, persist_directory='./data'):
        return Chroma.from_documents(
            documents,
            embedding=embedding,
            persist_directory='./data'
            )