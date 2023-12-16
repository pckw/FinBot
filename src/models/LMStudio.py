from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.llms import OpenAI
from openai import OpenAI as OpenAI_native
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
import json
load_dotenv('.env')


class LMStudio_assistant():
    def __init__(self, vectordb, model_name='gpt-3.5-turbo', temperature=0.7, k=6) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.k = k
        self.vectordb = vectordb
        self.chat_history = []
        self.retriever = vectordb.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
            )

    def query(self, query):
        # load json from file
        with open('prompt_templates/few_shot_doc_prompt_de_mistral.json', 'r') as f:
            messages = json.load(f)

        retrieved_docs = self.retriever.invoke(query)
        for doc in retrieved_docs:
            print(doc.page_content)
            print("---------------------------------------------------------------------------------")
        retrieved_docs = """\n\nContext:""".join([doc.page_content for doc in retrieved_docs])
        prompt = " ".join(
            [
                messages[0]['content'].split("Question")[0],
                "\n\nContext:",
                retrieved_docs,
                "\n\nQuestion:",
                query
            ]
        )
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": "Final Answer:"})
        llm = OpenAI_native(base_url="http://localhost:1234/v1", api_key="not-needed")
        response = llm.chat.completions.create(
            model="local-model",  # this field is currently unused
            messages=messages,
            temperature=0.0
        )
        #print(response.choices[0].message.content)
        return {"answer": response.choices[0].message.content}

class LMStudio_extractor():
    def __init__(self, vectordb) -> None:
        self.retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    def extract_single_entity(self, entity: str, description: str, llm) -> str:
        retrieved_docs = self.retriever.invoke(entity)
        system_message = f"""
            You are an expert in extracting information from text. You only use information included in the context. You only return the requested information. You answer as brief as possible and never in a full sentence. Answer with NA if the informatin is unknown.
            """
        fewshot_user1 = f"""
            Extract the following information from the context: Number of employees (The average number of employees working at the company)\n
            Context: Während des Berichtzeitraums waren im Mittel 13.2 Personen bei SuperBrew AG angestellt. Betriebsbedingte kündigungen haben nicht stattgefunden .\n
            Number of employees:
            """
        fewshot_assistant1 = "13.2"
        fewshot_user2 = f"""
            Extract the following information from the context: Managing Director (The names of the managing directors of the company)
            Context: Das Geschäftsjahr endete mit einem Plus von 12 Prozent. Verantwortliche Geschäftsführer waren Samuel Altmann und Thomas Moore.
            Managing director:
            """
        fewshot_assistant2 = "Samual Altmann, Thomas Moore"
        prompt_template = PromptTemplate.from_template(
            """Extract the following information from the context: {entity} ({description})\n
            Context:{context}\n\n
            {entity}:"""
        )
        prompt = prompt_template.format_prompt(entity=entity,
                                                description=description,
                                                context=[doc.page_content for doc in retrieved_docs][0])
        print(prompt)
        #prompt = prompt_template.format_prompt()
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": fewshot_user1},
            {"role": "assistant", "content": fewshot_assistant1},
            {"role": "user", "content": fewshot_user2},
            {"role": "assistant", "content": fewshot_assistant2},
            {"role": "user", "content": prompt.to_messages()[0].content},
            ]
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
