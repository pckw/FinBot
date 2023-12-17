from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.llms import OpenAI
from openai import OpenAI as OpenAI_native
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from src.utils.fill_prompt_template import fill_prompt_template
import json
import copy
load_dotenv('.env')


class LMStudio_assistant():
    def __init__(self, vectordb, model_name='gpt-3.5-turbo', temperature=0.7, k=6) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.k = k
        self.vectordb = vectordb
        self.retriever = vectordb.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
            )
        with open('prompt_templates/few_shot_doc_prompt_de_mistral.json', 'r') as f:
            messages = json.load(f)
        self.chat_history = messages[:-2]
        self.prompt_template = messages[-2:]

    def query(self, query):
        # Retrieve doctuments
        retrieved_docs = self.retriever.invoke(query)
        # for doc in retrieved_docs:
        #     print(doc.page_content)
        #     print("---------------------------------------------------------------------------------")
        retrieved_docs = """\n\nContext:""".join([doc.page_content for doc in retrieved_docs])
        # Construct prompt from template
        prompt = copy.deepcopy(self.prompt_template)
        prompt[0] = fill_prompt_template(
            prompt[0],
            {"context": retrieved_docs, "query": query})
        self.chat_history.extend(prompt)
        # Define llm
        llm = OpenAI_native(base_url="http://localhost:1234/v1", api_key="not-needed")
        # Invoke LLM
        response = llm.chat.completions.create(
            model="local-model",  # this field is currently unused
            messages=self.chat_history,
            temperature=0.0
        )
        # Append response to history
        self.chat_history[-1] = {
            'role': 'assistant', 'content': "Final Answer: " + response.choices[0].message.content
            }
        return {"answer": response.choices[0].message.content}

class LMStudio_extractor():
    def __init__(self, vectordb) -> None:
        self.retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def extract_single_entity(self, entity: str, description: str, llm) -> str:
        retrieved_docs = self.retriever.invoke(entity)
        retrieved_docs = ';'.join([doc.page_content for doc in retrieved_docs])
        with open('prompt_templates/few_shot_extractor_prompt_de_mistral.json', 'r') as f:
            messages = json.load(f)
        messages[-1] = fill_prompt_template(
            messages[-1],
            {"entity": entity, "description": description, "context": retrieved_docs})
        response = llm.chat.completions.create(model="local-model", # this field is currently unused
                                               messages=messages,
                                               temperature=0.0,
                                              )
        return response.choices[0].message.content

    def extract_entities(self, entities: dict) -> dict:
        llm = OpenAI_native(base_url="http://localhost:1234/v1", api_key="not-needed")
        result = {}
        for i in entities:
            result[i] = self.extract_single_entity(entity=i,
                                                   description=entities[i],
                                                   llm=llm)
        return result
