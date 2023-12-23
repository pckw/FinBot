from langchain.vectorstores import Chroma
import shutil
from time import sleep
import os

class chromaDB():
    def __init__(self) -> None:
        pass

    def create_vectordb(documents, embedding, persist_directory='./data'):
        shutil.rmtree(persist_directory, ignore_errors=True)
        vectordb = Chroma.from_documents(
            documents,
            embedding=embedding,
            persist_directory=persist_directory
            )
        vectordb.persist()
        return vectordb

    def read_vectordb(embedding, persist_directory='./data'):
        return Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory
        )
