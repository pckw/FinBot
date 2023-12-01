from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

class TextDataset():
    def __init__(self, path):
        self.path = path

    def load(self):
        # load the document
        loader = PyPDFLoader(self.path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(documents)
        return documents