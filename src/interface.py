from langchain.embeddings import OpenAIEmbeddings
#from langchain.embeddings.cohere import CohereEmbeddings
from src.models.chatGPT import chatGPT_assistant, chatGPT_extractor
from src.models.LMStudio import LMStudio_assistant, LMStudio_extractor
from src.models.cohere import cohere_assistant, cohere_extractor
#from src.vectordb.chroma import chromaDB
from src.vectordb.qdrant import qdrantDB
from src.TextDataset import TextDataset
import json
from time import sleep
from typing import Union


def run_single_emmbedding(
        file: str,
        persist_directory: str,
        chunk_size: int,
        chunk_overlap: int,
        api_key: Union[str, None]
):
    """
    Run a embedding on a file and load it into a vector database.

    Args:
        file (str): The path to the file containing the text data.
        persist_directory (str): The directory where the vector database should be persisted.
        chunk_size (int): The size of each chunk of text to process.
        chunk_overlap (int): The amount of overlap between adjacent chunks.
        api_key (str): The API key for accessing the embedding service.

    Returns:
        None
    """
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


def run_double_embedding(file: str, api_key: Union[str, None]) -> None:
    """
    Runs embedding processes for the databases used for chat and key property extract.
    
    Parameters:
        file (str): The path to the file to be processed.
        api_key (str): The API key to be used for the embedding process.
    
    Returns:
        None
    """
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


def read_vectordb(persist_directory: str, api_key: Union[str, None]) -> object:
    """
    Initializes an Embeddings object with the provided API key and reads a vector database.

    Parameters:
        persist_directory (str): The directory path where the vector database is persisted.
        api_key (Union[str, None]): The API key for the OpenAI service. Can be None if no API key is required.

    Returns:
        object: The vector database read from qdrantDB.
    """
    embedding = OpenAIEmbeddings(api_key=api_key)
    #embedding = CohereEmbeddings()
    persist_directory = persist_directory
    return qdrantDB.read_vectordb(
        embedding=embedding,
        persist_directory=persist_directory
    )


def get_response(message, chat_history: list, api_key: Union[str, None]) -> str:
    """
    This function takes in a message, a chat history, and an API key (which can be a string or None) as parameters.
    It retrieves the model parameters from a JSON file, including the model name, k value, and temperature.
    The function then reads the vectordb from a specified directory using the provided API key.
    Based on the model name, the function initializes the appropriate assistant object.
    The function queries the assistant with the given message and appends the message and the assistant's answer to the chat history.
    Finally, the function returns an empty string and the updated chat history.

    Parameters:
    - message (str): The message for the assistant to process.
    - chat_history (list): A list of tuples representing the chat history. Each tuple contains a message and the assistant's response.
    - api_key (Union[str, None]): The API key used to access the vectordb. It can be either a string or None.

    Returns:
    - tuple: An empty string and the updated chat history.

    """
    with open('./config/model_parameter.json', 'r') as f:
        config = json.load(f)
        model_name = config["model_name"]
        k = config["k_chat"]
        temperature = config["temperature"]
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


def key_properties(model_name: str, api_key: Union[str, None]) -> tuple:
    """
    Retrieves key properties from a given model using the provided model name and API key.
    
    Args:
        model_name (str): The name of the model.
        api_key (Union[str, None]): The API key to access the model. If None, no API key is required.
    
    Returns:
        tuple: A tuple containing the extracted key properties. The tuple includes the following elements:
            - Name (str): The name of the model.
            - Headquarters (str): The headquarters of the model.
            - Number of employees (int): The number of employees associated with the model.
            - Managing director (str): The name of the managing director of the model.
            - Report period (str): The report period for the model.
    """
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


def embeddings_and_key_properties(
        file: str,
        model_name: str,
        api_key: Union[str, None]
) -> tuple:
    """
    Generate the embeddings and key properties for the given file using the specified model 
    and API key.

    Args:
        file (str): The path to the file.
        model_name (str): The name of the model to use.
        api_key (Union[str, None]): The API key to use, or None if no API key is required.

    Returns:
        tuple: A tuple containing the name, headquarters, employee count, manager count, and 
        period of the key properties.
    """
    run_double_embedding(file, api_key)
    sleep(0.5)
    name, hq, employee, manager, period = key_properties(model_name, api_key)
    return name, hq, employee, manager, period
    #return "A", "B", "C", "D", "E"


def uploadbutton(file: str, model_name: str, api_key: Union[str, None]) -> tuple:
    """
    runs the embeddings_and_key_properties function for the given file and returns the name, headquarters, employee, manager, and period.

    Args:
        file (str): The path of the file to upload.
        model_name (str): The name of the model to use for processing the file.
        api_key (Union[str, None]): The API key to authenticate the upload. None if no authentication is required.

    Returns:
        tuple: A tuple containing the name, headquarters, employee, manager, and period.
            - name (str): The name extracted from the file.
            - hq (str): The headquarters extracted from the file.
            - employee (str): The employee extracted from the file.
            - manager (str): The manager extracted from the file.
            - period (str): The period extracted from the file.
    """
    if file:
        name, hq, employee, manager, period = embeddings_and_key_properties(
            file=file,
            model_name=model_name,
            api_key=api_key
        )
        return name, hq, employee, manager, period
    else:
        return "", "", "", "", ""


def read_parameter_from_file(file: str ='./config/model_parameter_default.json') -> dict:
    """
    Replace config with default_config
    Read the parameter from the file.

    Args:
        file (str): The path to the file to read the parameter from. Defaults to './config/model_parameter_default.json'.

    Returns:
        dict: The parameter read from the file.

    """
    with open(file, 'r') as f:
        config = json.load(f)
    # replace config with default_config
    with open(file, 'w') as f:
        json.dump(config, f)
    return config


def write_parameter_to_file(
        key: str,
        value: Union[int, str],
        file='./config/model_parameter.json'
) -> None:
    """
    Write the given key-value pair to a JSON file.

    Parameters:
        key (str): The key for the parameter.
        value (Union[int, str]): The value for the parameter.
        file (str): The path to the JSON file to write to. Default is './config/model_parameter.json'.

    Returns:
        None
    """
    with open(file, 'r') as f:
        config = json.load(f)
        config[key] = value
    with open(file, 'w') as f:
        json.dump(config, f)


def write_model_to_file(value: str) -> None:
    write_parameter_to_file('model', value)


def write_temperature_to_file(value: int) -> None:
    write_parameter_to_file('temperature', int(value))


def write_kchat_to_file(value: int) -> None:
    write_parameter_to_file('k_chat', int(value))


def write_kkeyprop_to_file(value: int) -> None:
    write_parameter_to_file('k_keyprop', int(value))


def write_chunkchat_to_file(value: int) -> None:
    write_parameter_to_file('chunk_chat', int(value))


def write_chunkkeyprop_to_file(value: int) -> None:
    write_parameter_to_file('chunk_keyprop', int(value))


def write_overlapchat_to_file(value: int) -> None:
    write_parameter_to_file('overlap_chat', int(value))


def write_overlapkeyprop_to_file(value: int) -> None:
    write_parameter_to_file('overlap_keyprop', int(value))


def read_default() -> tuple:
    """
    Reads the default model parameters from the 'model_parameter_default.json' file.

    Returns:
        tuple: A tuple containing the following model parameters:
            - model_name (str): The name of the model.
            - temperature (int): The temperature parameter.
            - k_chat (int): The k_chat parameter.
            - k_keyprop (int): The k_keyprop parameter.
            - chunk_chat (int): The chunk_chat parameter.
            - chunk_keyprop (int): The chunk_keyprop parameter.
            - overlap_chat (int): The overlap_chat parameter.
            - overlap_keyprop (int): The overlap_keyprop parameter.
    """
    with open('./config/model_parameter_default.json', 'r') as f:
        config = json.load(f)
    return\
        config['model_name'], \
        config['temperature'], \
        config['k_chat'], \
        config['k_keyprop'], \
        config['chunk_chat'], \
        config['chunk_keyprop'], \
        config['overlap_chat'], \
        config['overlap_keyprop']


def display_pdf(file):
    return file
