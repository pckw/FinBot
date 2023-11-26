from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

load_dotenv('.env')

class chatGPT_assistant():
    def __init__(self, vectordb, model_name='gpt-3.5-turbo', temperature=0.7, k=6) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.k = k
        self.vectordb = vectordb
    
    def query(self, query, chat_history=[]):

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            """Your name is FinBot and you are an expert on financial data and reports.
              Answer the user question based on the provided context delimited 
              by tripple backticks. Make your answer as brief and short as 
              possible. Answer with a single word or number when possible. 
              If you don't know the answer, say that you don't know. 
              Dont make up any answer.\n
              ``` Context: {context} ```""") 
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            "{question}")

        qa_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=self.temperature,
                        model_name=self.model_name),
            retriever=self.vectordb.as_retriever(search_kwargs={'k': self.k}),
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": ChatPromptTemplate.
                                       from_messages([system_message_prompt,
                                                      human_message_prompt])},
        )
        return qa_chain({"question": query, "chat_history": chat_history})