from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  #, CharacterTextSplitter


class TextDataset():
    def __init__(self, path: str) -> None:
        """
        Initializes an instance of the class.

        Args:
            path (str): The path to the file.

        Returns:
            None
        """
        self.path = path

    def load(self, chunk_size: int, chunk_overlap: int) -> list:
        """
        Load the document using a PyPDFLoader and split the text into chunks using a RecursiveCharacterTextSplitter.
        
        Parameters:
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The number of characters overlapping between adjacent chunks.
        
        Returns:
            list: A list of documents, each represented as a string.
        """
        # load the document
        loader = PyPDFLoader(self.path)
        documents = loader.load()
        #text_splitter = CharacterTextSplitter(separator='\n')
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            )
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
