from langchain.embeddings import OpenAIEmbeddings
from src.utils.get_files_from_directory import get_files_from_directory
import gradio as gr
from gradio_pdf import PDF
import yaml
import shutil
import os
from src.interface import read_parameter_from_file, get_response, \
    uploadbutton, write_model_to_file, \
    write_temperature_to_file, write_kchat_to_file, write_kkeyprop_to_file, \
    write_chunkchat_to_file, write_chunkkeyprop_to_file, \
    write_overlapchat_to_file, write_overlapkeyprop_to_file, \
    read_default, display_pdf


def main():
    try:
        with open("./config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            api_key_from_config = config["OPENAI_API_KEY"]
    except FileNotFoundError:
        api_key_from_config = None
    # check if model_parameter.json exists
    # and copy model_parameter_default.json otherwise
    if not os.path.exists('./config/model_parameter.json'):
        shutil.copy(
            './config/model_parameter_default.json',
            './config/model_parameter.json'
        )
    config = read_parameter_from_file()
    with gr.Blocks() as iface:
        gr.Markdown("# FinBot")
        # Chat tab
        with gr.Tab("Chat"):
            with gr.Row():
            # add another box next to inputfile_list
                inputfile_button = gr.File(
                    label="Upload a file",
                    file_types=[".pdf"],
                    container=True,
                    #variant="primary",
                    #size="sm",
                    scale=1,
                    type="filepath"
                )
            with gr.Row():
                with gr.Column(scale=6):
                    chat_history = gr.Chatbot()
                    msg = gr.Textbox(label="Input")
                    gr.ClearButton([msg, chat_history], value="Clear console")
                with gr.Column(scale=2):
                    gr.Markdown("### Key properties")
                    name = gr.Textbox(label="Name of the company")
                    period = gr.Textbox(label="Report period")
                    hq = gr.Textbox(label="Headquarter")
                    employee = gr.Textbox(label="Number of employee")
                    manager = gr.Textbox(label="Managing director(s)")
                    #revenue = gr.Textbox(label="Revenue/Loss")


        # Options tab
        with gr.Tab("Options"):
            model = gr.Dropdown(
                label="Model",
                value="gpt-3.5-turbo",
                choices=["gpt-3.5-turbo", "gpt-4", "Cohere", "LM Studio"],
                interactive=True
            )
            api_key = gr.Textbox(
                value=api_key_from_config,
                label="API Key",
                interactive=True,
                type="password"
                )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Chat parameter")
                    k_chat = gr.Textbox(
                        label="Number of retrieved chunks",
                        value=config["k_chat"],
                        interactive=True
                    )
                    chunk_size_chat = gr.Textbox(
                        label="Size of retrieved chunks",
                        value=config["chunk_chat"],
                        interactive=True
                    )
                    overlap_chat = gr.Textbox(
                        label="Overlap of retrieved chunks",
                        value=config["overlap_chat"],
                        interactive=True
                    )
                    temp = gr.Slider(label="Temperature", value=config["temperature"])
                with gr.Column():
                    gr.Markdown("### Key properties extraction parameter")
                    k_keyprop = gr.Textbox(
                        label="Number of retrieved chunks",
                        value=config["k_keyprop"],
                        interactive=True
                    )
                    chunk_size_keyprop = gr.Textbox(
                        label="Size of retrieved chunks",
                        value=config["chunk_keyprop"],
                        interactive=True
                    )
                    overlap_keyprop = gr.Textbox(
                        label="Overlap of retrieved chunks",
                        value=config["overlap_keyprop"],
                        interactive=True
                    )
            reset = gr.Button("Reset to default")
        with gr.Tab("PDF"):
            gr.Markdown("### PDF")
            pdf = PDF(
                label="Upload a PDF",
                interactive=False,
                height=1000
            )
            ## Define actions ##
            msg.submit(
                fn=get_response,
                inputs=[msg, chat_history, api_key],
                outputs=[msg, chat_history]
            )
            inputfile_button.change(
                fn=uploadbutton,
                inputs=[inputfile_button, model, api_key],
                outputs=[name, hq, employee, manager, period])
            model.change(
                fn=write_model_to_file,
                inputs=[model],
            )
            # api_key.change(
            #     fn=write_apikey_to_file,
            #     inputs=[model],
            # )
            temp.change(
                fn=write_temperature_to_file,
                inputs=[temp],
            )
            k_chat.change(
                fn=write_kchat_to_file,
                inputs=[k_chat],
            )
            k_keyprop.change(
                fn=write_kkeyprop_to_file,
                inputs=[k_keyprop],
            )
            chunk_size_chat.change(
                fn=write_chunkchat_to_file,
                inputs=[chunk_size_chat],
            )
            chunk_size_keyprop.change(
                fn=write_chunkkeyprop_to_file,
                inputs=[chunk_size_keyprop],
            )
            overlap_chat.change(
                fn=write_overlapchat_to_file,
                inputs=[overlap_chat],
            )
            overlap_keyprop.change(
                fn=write_overlapkeyprop_to_file,
                inputs=[overlap_keyprop],
            )
            reset.click(
                fn=read_default,
                inputs=None,
                outputs=[
                    model,
                    temp,
                    k_chat,
                    k_keyprop,
                    chunk_size_chat,
                    chunk_size_keyprop,
                    overlap_chat,
                    overlap_keyprop,
                ]
            )
        inputfile_button.change(
            fn=display_pdf,
            inputs=[inputfile_button],
            outputs=[pdf],
        )
    iface.launch()
if __name__ == "__main__":
    main()