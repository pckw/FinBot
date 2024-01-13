from openai import OpenAI as OpenAI_native
from dotenv import load_dotenv
from src.utils.fill_prompt_template import fill_prompt_template
import json
import copy


class LMStudio_assistant():
    def __init__(self, vectordb, temperature: int = 0, k: int = 3) -> None:
        """
        Initializes an instance of the class.

        Args:
            vectordb (object): A description of the parameter vectordb.
            temperature (int, optional): A description of the parameter temperature. Defaults to 0.
            k (int, optional): A description of the parameter k. Defaults to 3.

        Returns:
            None
        """
        self.temperature = temperature
        self.k = k
        self.vectordb = vectordb
        self.retriever = vectordb.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
            )
        with open('prompt_templates/few_shot_doc_prompt_de_mistral.json', 'r') as f:
            messages = json.load(f)
        self.chat_history = messages[:-2]
        self.prompt_template = messages[-2:]

    def query(self, query: str) -> dict:
        """
        Retrieves documents based on the given query, constructs a prompt from a template, invokes a language model, and returns the final answer.

        Parameters:
            query (str): The query string used to retrieve documents.

        Returns:
            dict: A dictionary containing the answer as a string.
        """
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
    def __init__(self, vectordb: object) -> None:
        """
        Initializes the class instance.

        Args:
            vectordb: A vectordb object.

        Returns:
            None
        """
        self.retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

    def extract_single_entity(self, entity: str, description: str, llm: object) -> str:
        """
        Extracts a single entity from the given `entity` and `description` using the local language model `llm`.

        Args:
            entity (str): The name of the entity to extract.
            description (str): The description of the entity to extract.
            llm: The local language model to use for extraction.

        Returns:
            str: The extracted content.

        """
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
        """
        Extracts entities from a dictionary of entities.

        Args:
            entities (dict): A dictionary containing the entities to be extracted.

        Returns:
            dict: A dictionary containing the extracted entities.

        Raises:
            None
        """
        llm = OpenAI_native(base_url="http://localhost:1234/v1", api_key="not-needed")
        result = {}
        for i in entities:
            result[i] = self.extract_single_entity(entity=i,
                                                   description=entities[i],
                                                   llm=llm)
        return result
