from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
load_dotenv('.env')


class chatGPT_assistant():
    def __init__(self, vectordb, model_name='gpt-3.5-turbo', temperature=0.7, k=6) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.k = k
        self.vectordb = vectordb
        self.chat_history = []
    
    def query(self, query):
        # open prompt templates
        with open("prompt_templates/few_shot_doc_prompt_de_chatgpt.txt") as f:
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
        llm_for_chat = ChatOpenAI(temperature=self.temperature,
                                  model=self.model_name)
        
        llm_for_doc_chain = OpenAI(temperature=0)

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
    
class chatGPT_extractor():
        def __init__(self, vectordb) -> None:
            self.retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})

        def extract_single_entity(self, entity:str, description:str, llm) -> str:
            retrieved_docs = self.retriever.invoke(entity)
            prompt_template = PromptTemplate.from_template(
                """Extract the following information from the context. Answer very briefly and as short as possible. Answer with NA if not found.\n
                {entity}: {description}\n
                Context:{context}\n\n
                
                {entity}:"""
            )
            prompt = prompt_template.format_prompt(entity=entity,
                                                   description=description,
                                                   context=[doc.page_content for doc in retrieved_docs])
            response = llm(prompt.to_messages()).content
            return response

        def extract_entities(self, entities: dict) -> dict:
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            result = {}
            for i in entities:
                result[i] = self.extract_single_entity(entity=i,
                                                      description=entities[i],
                                                      llm=llm)
            return result
