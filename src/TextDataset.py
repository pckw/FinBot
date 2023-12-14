from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter, CharacterTextSplitter

class TextDataset():
    def __init__(self, path):
        self.path = path

    def load(self):
        # load the document
        loader = PyPDFLoader(self.path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(separator='\n')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        # text_splitter = RecursiveCharacterTextSplitter(separators=['\nHandelsbilanz\n',
        #                                                            '\nAnhang\n',
        #                                                            '\nAllgemeine Angaben zum Jahresabschluss\n',
        #                                                            '\nEreignisse nach dem Bilanzstichtag\n',
        #                                                            '\nAngaben zur Bilanz\n',
        #                                                            '\nAngabe zu Restlaufzeitvermerken\n',
        #                                                            '\nNamen der Geschäftsführer\n'
        #                                                            '\n\n'],
        #                                                chunk_size=3000,
        #                                                chunk_overlap=200)
        documents = text_splitter.split_documents(documents)
        return documents