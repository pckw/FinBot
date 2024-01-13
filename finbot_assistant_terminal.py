import os
import yaml
import sys
from langchain.embeddings import OpenAIEmbeddings
import json
from src.models.chatGPT import chatGPT_assistant, chatGPT_extractor
from src.models.LMStudio import LMStudio_assistant, LMStudio_extractor
from src.vectordb.chroma import chromaDB
from src.vectordb.qdrant import qdrantDB
from src.TextDataset import TextDataset


# Load documents
file = "./docs/Kiels_Fitness_2021.pdf"

with open('./config/key_properties.json', 'r') as f:
    key_properties = json.load(f)

try:
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        api_key = config["OPENAI_API_KEY"]
except FileNotFoundError:
    api_key = None

def create_vectordb(file, persist_directory, chunk_size, chunk_overlap):
    documents = TextDataset(file).load(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # Initialize the vector database
    embedding = OpenAIEmbeddings(api_key=api_key)
    vectordb = qdrantDB.create_vectordb(
        documents,
        embedding,
        persist_directory=persist_directory
    )
    # for d in documents:
    #     #print(d.page_content)
    #     print(d)
    #     print("---------------------------------------------------------------------------------")
    # print("---------------------------------------------------------------------------------")
    # print("---------------------------------------------------------------------------------")
    # exit()
    return vectordb

vectordb_chat = create_vectordb(
    file=file,
    persist_directory='./data/chat',
    chunk_size=1500,
    chunk_overlap=250,
    )

vectordb_keyprop = create_vectordb(
    file=file,
    persist_directory='./data/keyprop',
    chunk_size=500,
    chunk_overlap=0,
    )

# Initialize the assistant
model_name = 'gpt-3.5-turbo'
#model_name = 'gpt-4'
finbot_assistant = chatGPT_assistant(vectordb=vectordb_chat,
                                     model_name=model_name,
                                     temperature=0,
                                     k=3,
                                     api_key=api_key)
# finbot_assistant = LMStudio_assistant(vectordb=vectordb_chat,
#                                      model_name=model_name,
#                                      temperature=0,
#                                      k=3)

extractor = chatGPT_extractor(vectordb=vectordb_keyprop, api_key=api_key)
#extractor = LMStudio_extractor(vectordb=vectordb_keyprop)
extracted_key_properties = extractor.extract_entities(entities=key_properties)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the FinBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
print(' ')
print('Derived key properties:')
for i in extracted_key_properties:
    print(f"{i}: {extracted_key_properties[i]}")
print(' ')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue

    result = finbot_assistant.query(query)
    print(f"{white}Answer: " + result["answer"])
