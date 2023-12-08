from langchain.embeddings import OpenAIEmbeddings
from src.models.chatGPT import chatGPT_assistant, chatGPT_extractor
from src.vectordb.chroma import chromaDB
from src.TextDataset import TextDataset
import gradio as gr
import json


def run_emmbedding(file):
    documents = TextDataset(file).load()
    embedding = OpenAIEmbeddings()
    vectordb = chromaDB.create_vectordb(documents,
                                        embedding,
                                        persist_directory='./data')
    vectordb.persist()

def read_vectordb():
    embedding = OpenAIEmbeddings()
    persist_directory = './data'
    return chromaDB.read_vectordb(embedding=embedding, persist_directory=persist_directory)


def get_response(message, chat_history, model_name, temperature):
    #load_dotenv('.env')
    vectordb = read_vectordb()
    finbot_assistant = chatGPT_assistant(vectordb=vectordb,
                                         model_name=model_name,
                                         temperature=temperature,
                                         k=3)
    result = finbot_assistant.query(message)
    chat_history.append((message, result["answer"]))
    return "", chat_history


def update_key_properties():
    with open('key_properties.json','r') as f:
        key_properties = json.load(f)
    vectordb = read_vectordb()
    key_property_extractor = chatGPT_extractor(vectordb=vectordb)
    extracted_key_properties = key_property_extractor.extract_entities(entities=key_properties)
    print(extracted_key_properties)
    return extracted_key_properties['Name'],\
           extracted_key_properties['Headquarters'],\
           extracted_key_properties['Number of employees']


def main():
    with gr.Blocks() as iface:
        gr.Markdown("# FinBot")
        with gr.Row():
            with gr.Column(scale=6):
                with gr.Row():
                    with gr.Column(scale=4):
                        file = gr.Textbox(label="PDF File", value="./docs/Lillebräu_2021.pdf")
                    with gr.Column(scale=1, min_width=100):
                        upload = gr.Button(value="Upload")            
                chat_history = gr.Chatbot()
                msg = gr.Textbox(label="Input")
                gr.ClearButton([msg, chat_history], value="Clear console")
            with gr.Column(scale=2):
                model = gr.Dropdown(label="Model",
                                    value="gpt-3.5-turbo",
                                    choices=["gpt-3.5-turbo", "gpt-4"])
                api_key = gr.Textbox(label="API Key")
                temp = gr.Slider(label="Temperature", value=0)
                gr.Markdown("### Key properties")
                name = gr.Textbox(label="Name of the company")
                hq = gr.Textbox(label="Headquarter")
                employee = gr.Textbox(label="Number of employee")
                update = gr.Button(value="Update key properties")
                #manager = gr.Textbox(label="Managing director(s)")
                #revenue = gr.Textbox(label="Revenue/Loss")
        msg.submit(fn=get_response,
                   inputs=[msg, chat_history, model, temp],
                   outputs=[msg, chat_history])
        upload.click(fn=run_emmbedding, inputs=file)
        update.click(fn=update_key_properties,
                     inputs=[],
                     outputs=[name, hq, employee])
    iface.launch()


if __name__ == "__main__":
    main()