from langchain.chains import ConversationalRetrievalChain, LLMChain
#from langchain.llms import OpenAI
from langchain.chat_models import ChatCohere
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
import yaml
from typing import Union


class cohere_assistant():
    def __init__(
            self,
            vectordb: object, 
            temperature: int = 0,
            k: int = 3,
            api_key: Union[str, None] = None
    ) -> None:
        """
        Initializes the object with the given `vectordb`, `temperature`, `k`, and `api_key`.

        Parameters:
            vectordb (object): The vectordb to be used for initialization.
            temperature (int): The temperature value to be set. Default is 0.
            k (int): The k value to be set. Default is 3.
            api_key (str): The API key to be used. If not provided, it will be read from `config.yaml`.
     
        Returns:
            None
        """
        self.temperature = temperature
        self.k = k
        self.vectordb = vectordb
        self.chat_history = []
        # read cohere api key from config.yaml
        if api_key:
            self.api_key = api_key
        else:
            with open("./config/config.yaml", "r") as f:
                config = yaml.safe_load(f)
                self.api_key = config["COHERE_API_KEY"]
    
    def query(self, query: str) -> dict:
        """
        Executes a query using the provided `query` parameter and returns the result.

        Parameters:
            query (str): The query to be executed.

        Returns:
            dict: The result of the query execution.

        Raises:
            FileNotFoundError: If any of the prompt template files cannot be found.
        """
        # open prompt templates
        with open("prompt_templates/few_shot_doc_prompt_de_cohere.txt") as f:
            template_few_shot_doc = f.read()

        with open("prompt_templates/summary_en.txt") as f:
            template_summary = f.read()
        # build prompts
        SUMMARY_PROMPT = PromptTemplate.from_template(template_summary)

        QA_PROMPT = PromptTemplate(template=template_few_shot_doc,
                                   input_variables=["summaries", "question"])
        DOC_PROMPT = PromptTemplate(
            template="Content: {page_content}",
            input_variables=["page_content"])

        # Define llms
        llm_for_chat = ChatCohere(
            temperature=self.temperature,
            cohere_api_key=self.api_key
        )
       
        llm_for_doc_chain = ChatCohere(
            temperature=0,
            cohere_api_key=self.api_key
        )

        # Define chains
        question_generator = LLMChain(llm=llm_for_doc_chain,
                                      prompt=SUMMARY_PROMPT)
        
        doc_chain = load_qa_with_sources_chain(llm_for_chat,
                                               chain_type="stuff",
                                               verbose=True,
                                               prompt=QA_PROMPT,
                                               document_prompt=DOC_PROMPT)
        
        chain = ConversationalRetrievalChain(
                retriever=self.vectordb.as_retriever(search_type="mmr", search_kwargs={'k': self.k}),
                question_generator=question_generator,
                combine_docs_chain=doc_chain,
                rephrase_question=False,
                return_source_documents=False,
                return_generated_question=False,
                verbose=True
            )
        result = chain({"question": query, "chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"])) # add query and result
        return result


class cohere_extractor():
    def __init__(self, vectordb, api_key) -> None:
        """
        Initializes an instance of the class.

        Parameters:
            vectordb (object): The vectordb object.
            api_key (str): The API key.

        Returns:
            None
        """
        self.retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        if api_key:
            self.api_key = api_key
        else:
            with open("./config/config.yaml", "r") as f:
                config = yaml.safe_load(f)
                self.api_key = config["COHERE_API_KEY"]

    def extract_single_entity(self, entity: str, description: str, llm: object) -> str:
        """
        Extracts a single entity from the given context using the specified entity and description.
        
        Parameters:
            entity (str): The entity to extract from the context.
            description (str): The description of the entity.
            llm: The language model.
        
        Returns:
            str: The extracted entity.
        """
        retrieved_docs = self.retriever.invoke(entity)
        for doc in retrieved_docs:
            print(entity)
            print(doc.page_content)
            print("---------------------------------------------------------------------------------")
        prompt_template = PromptTemplate.from_template(
            """Extract the following information from the context. Answer as short as possible. Answer with NA if not found and rember to answer with as few words as possible. Don't offer do extract more information.\n
            Revenue: The revenue of the company in the last year.
            Context: "Total revenue in the last year was $10 million."

            Revenue: $10 million
            
            {entity}: {description}\n
            Context:{context}\n\n
            
            {entity}:"""
        )
        prompt = prompt_template.format_prompt(
            entity=entity,
            description=description,
            context="\n".join([doc.page_content for doc in retrieved_docs])
        )
        response = llm(prompt.to_messages()).content
        return response

    def extract_entities(self, entities: dict) -> dict:
        """
        Extracts entities from a dictionary and returns a dictionary with the extracted entities.

        Args:
            entities (dict): A dictionary containing the entities to extract.

        Returns:
            dict: A dictionary with the extracted entities.
        """
        llm = ChatCohere(temperature=0, cohere_api_key=self.api_key)
        result = {}
        for i in entities:
            result[i] = self.extract_single_entity(entity=i,
                                                    description=entities[i],
                                                    llm=llm)
        return result
