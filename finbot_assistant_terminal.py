import os
import sys
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import json
from src.models.chatGPT import chatGPT_assistant, chatGPT_extractor
from src.models.LMStudio import LMStudio_assistant, LMStudio_extractor
from src.vectordb.chroma import chromaDB
from src.TextDataset import TextDataset
from src.utils.get_source_pdf_from_directory import get_source_pdf_from_directory

load_dotenv('.env')

# Load documents
source_directory = "./docs/Lillebraeu_2021"
#source_directory = "./docs/Kieler_Brauerei_2021"

with open('key_properties.json','r') as f:
    key_properties = json.load(f)

def create_vectordb(source_directory, persist_directory, chunk_size, chunk_overlap):
    file = get_source_pdf_from_directory(source_directory)
    documents = TextDataset(file).load(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Initialize the vector database
    embedding = OpenAIEmbeddings()
    vectordb = chromaDB.create_vectordb(documents,
                                embedding,
                                persist_directory=source_directory+"/data/"+persist_directory)
    vectordb.persist()
    # for d in documents:
    #     #print(d.page_content)
    #     print(d)
    #     print("---------------------------------------------------------------------------------")
    # print("---------------------------------------------------------------------------------")
    # print("---------------------------------------------------------------------------------")
    # exit()
    return vectordb

vectordb_chat = create_vectordb(
    source_directory=source_directory,
    persist_directory='chat',
    chunk_size=2000,
    chunk_overlap=200,
    )

vectordb_keyprop = create_vectordb(
    source_directory=source_directory,
    persist_directory='keyprop',
    chunk_size=300,
    chunk_overlap=0,
    )
#vectordb = chromaDB.read_vectordb(embedding, persist_directory=source_directory)

# query it
# query = "Wie heißen die Geschäftsführer?"
# retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={"k": 3})
# docs=retriever.get_relevant_documents(query)

# Initialize the assistant
model_name='gpt-3.5-turbo'
#model_name='gpt-4'
# finbot_assistant = chatGPT_assistant(vectordb=vectordb_chat,
#                                      model_name=model_name,
#                                      temperature=0,
#                                      k=3)
finbot_assistant = LMStudio_assistant(vectordb=vectordb_chat,
                                     model_name=model_name,
                                     temperature=0,
                                     k=3)

#extractor = chatGPT_extractor(vectordb=vectordb_keyprop)
#extractor = LMStudio_extractor(vectordb=vectordb_keyprop)
#extracted_key_properties = extractor.extract_entities(entities=key_properties)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the FinBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
print(' ')
# print('Derived key properties:')
# for i in extracted_key_properties:
#     print(f"{i}: {extracted_key_properties[i]}")
# print(' ')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue

    result = finbot_assistant.query(query)
    print(f"{white}Answer: " + result["answer"])
