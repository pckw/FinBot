from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from src.models.chatGPT import chatGPT_assistant, chatGPT_extractor
from src.models.LMStudio import LMStudio_assistant, LMStudio_extractor
from src.models.cohere import cohere_assistant, cohere_extractor
from src.vectordb.chroma import chromaDB
from src.TextDataset import TextDataset
from src.utils.get_source_pdf_from_directory import get_source_pdf_from_directory
from src.utils.get_files_from_directory import get_files_from_directory
import gradio as gr
import json
from time import sleep


def run_single_emmbedding(file, persist_directory, chunk_size, chunk_overlap):
    documents = TextDataset(file).load(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embedding = OpenAIEmbeddings()
    #embedding = CohereEmbeddings()
    _ = chromaDB.create_vectordb(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory,
    )


def run_double_embedding(file):
    run_single_emmbedding(
        file=file,
        persist_directory='./data/keyprop',
        chunk_size=300,
        chunk_overlap=0,
    )
    run_single_emmbedding(
        file=file,
        persist_directory='./data/chat',
        chunk_size=1500,
        chunk_overlap=200
    )


def read_vectordb(persist_directory):
    embedding = OpenAIEmbeddings()
    #embedding = CohereEmbeddings()
    persist_directory = persist_directory
    return chromaDB.read_vectordb(
        embedding=embedding,
        persist_directory=persist_directory
    )


def get_response(message, chat_history, model_name, temperature):
    vectordb = read_vectordb(
        persist_directory='./data/chat')
    if model_name == 'LM Studio':
        finbot_assistant = LMStudio_assistant(vectordb=vectordb,
                                            temperature=temperature,
                                            k=3)
    elif model_name == 'gpt-4' or model_name == 'gpt-3.5-turbo':
        finbot_assistant = chatGPT_assistant(
            vectordb=vectordb,
            model_name=model_name,
            temperature=temperature,
            k=3
        )
    elif model_name == 'Cohere':
        finbot_assistant = cohere_assistant(
            vectordb=vectordb,
            temperature=temperature,
            k=3
        )
    result = finbot_assistant.query(message)
    chat_history.append((message, result["answer"]))
    return "", chat_history

def key_properties(model_name):
    with open('key_properties.json','r') as f:
        key_properties = json.load(f)
    vectordb = read_vectordb(
        persist_directory='./data/keyprop'
    )
    if model_name == 'LM Studio':
        key_property_extractor = LMStudio_extractor(vectordb=vectordb)
    elif model_name == 'gpt-4' or model_name == 'gpt-3.5-turbo':
        key_property_extractor = chatGPT_extractor(vectordb=vectordb)
    elif model_name == 'Cohere':
        key_property_extractor = cohere_extractor(vectordb=vectordb)
    extracted_key_properties = key_property_extractor.extract_entities(entities=key_properties)
    return extracted_key_properties['Name'],\
           extracted_key_properties['Headquarters'],\
           extracted_key_properties['Number of employees'],\
           extracted_key_properties['Managing directors']


def embeddings_and_key_properties(file, model_name):
    file = "./docs/"+file
    run_double_embedding(file)
    sleep(0.5)
    name, hq, employee, manager = key_properties(model_name)
    return name, hq, employee, manager
    #return "A", "B", "C", "D"


def main():
    list_of_files=get_files_from_directory('./docs')
    with gr.Blocks() as iface:
        gr.Markdown("# FinBot")
        with gr.Row():
            with gr.Column(scale=6):
                with gr.Row():
                    with gr.Column():
                        inputfile = gr.Dropdown(label="Input file", choices=list_of_files)
                chat_history = gr.Chatbot()
                msg = gr.Textbox(label="Input")
                gr.ClearButton([msg, chat_history], value="Clear console")
            with gr.Column(scale=2):
                model = gr.Dropdown(label="Model",
                                    value="gpt-3.5-turbo",
                                    choices=["gpt-3.5-turbo", "gpt-4", "Cohere", "LM Studio"])
                api_key = gr.Textbox(label="API Key")
                temp = gr.Slider(label="Temperature", value=0)
                gr.Markdown("### Key properties")
                name = gr.Textbox(label="Name of the company")
                hq = gr.Textbox(label="Headquarter")
                employee = gr.Textbox(label="Number of employee")
                manager = gr.Textbox(label="Managing director(s)")
                #revenue = gr.Textbox(label="Revenue/Loss")
        msg.submit(fn=get_response,
                   inputs=[msg, chat_history, model, temp],
                   outputs=[msg, chat_history])
        inputfile.change(
            fn=embeddings_and_key_properties,
            inputs=[inputfile, model],
            outputs=[name, hq, employee, manager]
        )
    iface.launch()


if __name__ == "__main__":
    main()