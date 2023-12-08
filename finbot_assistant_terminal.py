import os
import sys
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

from src.models.chatGPT import chatGPT_assistant
from src.vectordb.chroma import chromaDB
from src.TextDataset import TextDataset

load_dotenv('.env')

# Load documents
file = "./docs/Lillebräu_2021.pdf"
documents = TextDataset(file).load()

# print(len(documents))
# print(documents[0])
# exit()
# Initialize vecordb
embedding = OpenAIEmbeddings()
# vectordb = chromaDB.create_vectordb(documents,
#                              embedding,
#                              persist_directory='./data')
# vectordb.persist()
vectordb = chromaDB.read_vectordb(embedding, persist_directory='./data')

# Initialize the assistant
model_name='gpt-3.5-turbo'
#model_name='gpt-4'
finbot_assistant = chatGPT_assistant(vectordb=vectordb,
                                     model_name=model_name,
                                     temperature=0,
                                     k=3)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the FinBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = finbot_assistant.query(query)
    print(f"{white}Answer: " + result["answer"])
    #chat_history.append((query, result["answer"]))