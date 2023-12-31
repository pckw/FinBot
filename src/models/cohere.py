from langchain.chains import ConversationalRetrievalChain, LLMChain
#from langchain.llms import OpenAI
from langchain.chat_models import ChatCohere
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
import yaml


class cohere_assistant():
    def __init__(self, vectordb, temperature=0, k=3) -> None:
        #self.model_name = model_name
        self.temperature = temperature
        self.k = k
        self.vectordb = vectordb
        self.chat_history = []
        # read cohere api key from config.yaml
        with open("./config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            self.api_key = config["COHERE_API_KEY"]
    
    def query(self, query):
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
            template="Content: {page_content}\nSource: {source}",
            input_variables=["page_content", "source"])

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
        def __init__(self, vectordb) -> None:
            self.retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            with open("config.yaml", "r", encoding="UTF-8") as f:
                config = yaml.safe_load(f)
                self.api_key = config["COHERE_API_KEY"]

        def extract_single_entity(self, entity: str, description:str, llm) -> str:
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
            llm = ChatCohere(temperature=0, cohere_api_key=self.api_key)
            result = {}
            for i in entities:
                result[i] = self.extract_single_entity(entity=i,
                                                      description=entities[i],
                                                      llm=llm)
            return result
