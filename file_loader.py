import os
import re
import shutil 
from pathlib import Path

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document


# from langchain_community.document_loaders import GoogleSpeechToTextLoader


from langchain_community.document_loaders import(
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader
)

# Create file and loader mapping according to diffenent extensions

FILE_LOADER_MAPPING = {
    ".csv":(CSVLoader,{"encoding":"utf-8"}),
    ".xlsx": (UnstructuredExcelLoader,{}),
    ".doc":(UnstructuredWordDocumentLoader,{}),
    ".docx":(Docx2txtLoader,{}),
    ".html":(UnstructuredHTMLLoader,{}),
    ".md": (UnstructuredMarkdownLoader,{}),
    ".pdf":(PyPDFLoader,{}),
    ".ppt":(UnstructuredPowerPointLoader,{}),
    ".pptx":(UnstructuredPowerPointLoader,{}),
    ".txt":(TextLoader,{}),
    #You can add more mapping for other file extensions and loaders as needed
}


def load_document(folder_path:str, 
    mapping: dict = FILE_LOADER_MAPPING, 
    default_loader:BaseLoader = UnstructuredFileLoader
    ) -> Document:

    loaded_documents = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
    # choose loader from mapping as per file extension, load default if no matching loader is found.
        ext = "." + file_path.rsplit(".", 1)[-1].lower()
        # print("\n\n\n\n  file_path: ",file_path )
        if ext in mapping:
            
            loader_class, loader_args =mapping[ext]

            loader_args = {}  # Example: {"page_range": (1, 2)} for PDFs
            loader = loader_class(file_path, **loader_args)

            loaded_documents.extend(loader.load())
            
        
        else: 
            # loader = default_loader(file_path)
            loaded_documents.append(loader.load())


    return loaded_documents

