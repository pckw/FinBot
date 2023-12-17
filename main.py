from langchain.embeddings import OpenAIEmbeddings
from src.models.chatGPT import chatGPT_assistant, chatGPT_extractor
from src.models.LMStudio import LMStudio_assistant, LMStudio_extractor
from src.vectordb.chroma import chromaDB
from src.TextDataset import TextDataset
from src.utils.get_source_pdf_from_directory import get_source_pdf_from_directory
from src.utils.get_folders_from_directory import get_folders_from_directory
import gradio as gr
import json


def run_single_emmbedding(source_directory, persist_directory, chunk_size, chunk_overlap):
    file = get_source_pdf_from_directory(source_directory)
    persist_directory = source_directory+'/data/'+persist_directory
    documents = TextDataset(file).load(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    embedding = OpenAIEmbeddings()
    vectordb = chromaDB.create_vectordb(documents,
                                        embedding,
                                        persist_directory=persist_directory)
    vectordb.persist()

def run_double_embedding(source_directory):
    run_single_emmbedding(
        source_directory=source_directory,
        persist_directory='chat',
        chunk_size=2000,
        chunk_overlap=200,
    )
    run_single_emmbedding(
        source_directory=source_directory,
        persist_directory='keyprop',
        chunk_size=300,
        chunk_overlap=0,
    )

def read_vectordb(source_directory, persist_directory):
    embedding = OpenAIEmbeddings()
    persist_directory = source_directory+'/data/'+persist_directory
    return chromaDB.read_vectordb(
        embedding=embedding,
        persist_directory=persist_directory
    )


def get_response(message, chat_history, model_name, temperature, source_directory):
    #load_dotenv('.env')
    vectordb = read_vectordb(
        source_directory=source_directory,
        persist_directory='chat')
    if model_name == 'LM Studio':
        finbot_assistant = LMStudio_assistant(vectordb=vectordb,
                                            temperature=temperature,
                                            k=3)
    elif model_name == 'gpt-4' or model_name == 'gpt-3.5-turbo':
        finbot_assistant = chatGPT_assistant(vectordb=vectordb,
                                            model_name=model_name,
                                            temperature=temperature,
                                            k=3)
    result = finbot_assistant.query(message)
    chat_history.append((message, result["answer"]))
    return "", chat_history


def update_key_properties(source_directory, model_name):
    with open('key_properties.json','r') as f:
        key_properties = json.load(f)
    vectordb = read_vectordb(
        source_directory=source_directory,
        persist_directory='keyprop'
        )
    if model_name == 'LM Studio':
        key_property_extractor = LMStudio_extractor(vectordb=vectordb)
    else:
        key_property_extractor = chatGPT_extractor(vectordb=vectordb)
    extracted_key_properties = key_property_extractor.extract_entities(entities=key_properties)
    print(extracted_key_properties)
    return extracted_key_properties['Name'],\
           extracted_key_properties['Headquarters'],\
           extracted_key_properties['Number of employees'],\
           extracted_key_properties['Managing directors']


def main():
    list_of_source_folders=get_folders_from_directory('./docs')
    with gr.Blocks() as iface:
        gr.Markdown("# FinBot")
        with gr.Row():
            with gr.Column(scale=6):
                with gr.Row():
                    with gr.Column():
                        source_directory = gr.Dropdown(label="Source directory", choices=list_of_source_folders)
                chat_history = gr.Chatbot()
                msg = gr.Textbox(label="Input")
                gr.ClearButton([msg, chat_history], value="Clear console")
            with gr.Column(scale=2):
                model = gr.Dropdown(label="Model",
                                    value="gpt-3.5-turbo",
                                    choices=["gpt-3.5-turbo", "gpt-4", "LM Studio"])
                api_key = gr.Textbox(label="API Key")
                temp = gr.Slider(label="Temperature", value=0)
                gr.Markdown("### Key properties")
                name = gr.Textbox(label="Name of the company")
                hq = gr.Textbox(label="Headquarter")
                employee = gr.Textbox(label="Number of employee")
                manager = gr.Textbox(label="Managing director(s)")
                #revenue = gr.Textbox(label="Revenue/Loss")
        msg.submit(fn=get_response,
                   inputs=[msg, chat_history, model, temp, source_directory],
                   outputs=[msg, chat_history])
        source_directory.change(
            fn=run_double_embedding, inputs=[source_directory]
            )
        source_directory.change(
            fn=update_key_properties,
            inputs=[source_directory, model],
            outputs=[name, hq, employee, manager]
            )
    iface.launch()


if __name__ == "__main__":
    main()