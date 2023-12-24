from langchain.vectorstores import Chroma
import os

class chromaDB():
    def __init__(self) -> None:
        pass

    def create_vectordb(documents, embedding, persist_directory):
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

    def read_vectordb(embedding, persist_directory='./data'):
        return Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory
        )
