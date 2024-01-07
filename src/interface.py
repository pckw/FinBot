from langchain.embeddings import OpenAIEmbeddings
#from langchain.embeddings.cohere import CohereEmbeddings
from src.models.chatGPT import chatGPT_assistant, chatGPT_extractor
from src.models.LMStudio import LMStudio_assistant, LMStudio_extractor
from src.models.cohere import cohere_assistant, cohere_extractor
#from src.vectordb.chroma import chromaDB
from src.vectordb.qdrant import qdrantDB
from src.TextDataset import TextDataset
from src.utils.get_files_from_directory import get_files_from_directory
# import gradio as gr
# from gradio_pdf import PDF
import json
from time import sleep
# import yaml
# import shutil
# import os


def run_single_emmbedding(
        file,
        persist_directory,
        chunk_size,
        chunk_overlap,
        api_key
):
    documents = TextDataset(file).load(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embedding = OpenAIEmbeddings(api_key=api_key)
    #embedding = CohereEmbeddings()
    _ = qdrantDB.create_vectordb(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory,
    )


def run_double_embedding(file, api_key):
    run_single_emmbedding(
        file=file,
        persist_directory='./data/keyprop',
        chunk_size=600,
        chunk_overlap=0,
        api_key=api_key
    )
    run_single_emmbedding(
        file=file,
        persist_directory='./data/chat',
        chunk_size=1500,
        chunk_overlap=200,
        api_key=api_key
    )


def read_vectordb(persist_directory, api_key):
    embedding = OpenAIEmbeddings(api_key=api_key)
    #embedding = CohereEmbeddings()
    persist_directory = persist_directory
    return qdrantDB.read_vectordb(
        embedding=embedding,
        persist_directory=persist_directory
    )


def get_response(message, chat_history, api_key=None):
    with open('./config/model_parameter.json', 'r') as f:
        config = json.load(f)
        model_name = config["model_name"]
        k = config["k_chat"]
        temperature = config["temperature"]
        api_key = config["api_key"]
        if api_key == "":
            api_key = None
    vectordb = read_vectordb(
        persist_directory='./data/chat',
        api_key=api_key
    )
    if model_name == 'LM Studio':
        finbot_assistant = LMStudio_assistant(
            vectordb=vectordb,
            temperature=temperature,
            k=k
        )
    elif model_name == 'gpt-4' or model_name == 'gpt-3.5-turbo':
        finbot_assistant = chatGPT_assistant(
            vectordb=vectordb,
            model_name=model_name,
            temperature=temperature,
            k=k,
            api_key=api_key
        )
    elif model_name == 'Cohere':
        finbot_assistant = cohere_assistant(
            vectordb=vectordb,
            temperature=temperature,
            k=k,
            api_key=api_key
        )
    result = finbot_assistant.query(message)
    chat_history.append((message, result["answer"]))
    return "", chat_history


def key_properties(model_name, api_key):
    with open('./config/model_parameter.json', 'r') as f:
        config = json.load(f)
        k = config["k_keyprop"]
    with open('./config/key_properties.json', 'r') as f:
        key_properties = json.load(f)
    vectordb = read_vectordb(
        persist_directory='./data/keyprop',
        api_key=api_key
    )
    if model_name == 'LM Studio':
        key_property_extractor = LMStudio_extractor(vectordb=vectordb)
    elif model_name == 'gpt-4' or model_name == 'gpt-3.5-turbo':
        print("SETTING UP CHATGPT")
        key_property_extractor = chatGPT_extractor(
            vectordb=vectordb,
            api_key=api_key
        )
    elif model_name == 'Cohere':
        key_property_extractor = cohere_extractor(
            vectordb=vectordb,
            api_key=api_key
        )
    extracted_key_properties = key_property_extractor.extract_entities(
        entities=key_properties
    )
    return extracted_key_properties['Name'], \
        extracted_key_properties['Headquarters'], \
        extracted_key_properties['Number of employees'], \
        extracted_key_properties['Managing director'], \
        extracted_key_properties['Report period']


def embeddings_and_key_properties(file, model_name, api_key):
    run_double_embedding(file, api_key)
    sleep(0.5)
    name, hq, employee, manager, period = key_properties(model_name, api_key)
    return name, hq, employee, manager, period
    #return "A", "B", "C", "D", "E"


def uploadbutton(file, model_name, api_key=None):
    with open('./config/model_parameter.json', 'r') as f:
        config = json.load(f)
        api_key = config["api_key"]
        if api_key == "":
            api_key = None
    if file:
        name, hq, employee, manager, period = embeddings_and_key_properties(
            file=file,
            model_name=model_name,
            api_key=api_key
        )
        return name, hq, employee, manager, period
    else:
        return "", "", "", "", ""


def read_parameter_from_file(file='./config/model_parameter_default.json'):
    with open(file, 'r') as f:
        config = json.load(f)
    # replace config with default_config
    with open(file, 'w') as f:
        json.dump(config, f)
    return config


def write_parameter_to_file(key, value, file='./config/model_parameter.json'):
    with open(file, 'r') as f:
        config = json.load(f)
        config[key] = value
    with open(file, 'w') as f:
        json.dump(config, f)


def write_model_to_file(value):
    write_parameter_to_file('model', value)


def write_apikey_to_file(value):
    write_parameter_to_file('api_key', value)

def write_temperature_to_file(value):
    write_parameter_to_file('temperature', value)


def write_kchat_to_file(value):
    write_parameter_to_file('k_chat', value)


def write_kkeyprop_to_file(value):
    write_parameter_to_file('k_keyprop', value)


def write_chunkchat_to_file(value):
    write_parameter_to_file('chunk_chat', value)


def write_chunkkeyprop_to_file(value):
    write_parameter_to_file('chunk_keyprop', value)


def write_overlapchat_to_file(value):
    write_parameter_to_file('overlap_chat', value)


def write_overlapkeyprop_to_file(value):
    write_parameter_to_file('overlap_keyprop', value)


def read_default():
    with open('./config/model_parameter_default.json', 'r') as f:
        config = json.load(f)
    return\
        config['model_name'],\
        config['api_key'],\
        config['temperature'],\
        config['k_chat'],\
        config['k_keyprop'],\
        config['chunk_chat'],\
        config['chunk_keyprop'],\
        config['overlap_chat'],\
        config['overlap_keyprop']


def display_pdf(file):
    return file
