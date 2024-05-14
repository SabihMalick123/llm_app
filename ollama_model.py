from file_loader import load_document

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from langchain_community.vectorstores import Qdrant

# Load all document prsent in the given folder path 
def document_loader(folder_path):

    # Load documents
    documents = load_document(folder_path)

    # text splitter: Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    all_splits = (text_splitter.split_documents(documents))

    # print("\n\n all_splits", all_splits)

    return all_splits



def embedding_model():

    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    return embeddings


# Retrival : it takes documents and embedding model and create vector db. 
# Store embeddings in Chorma DB and return a retriever
def vector_db(all_splits, embeddings, persist_directory):

    qdrant = Qdrant.from_documents(
    all_splits,
    embeddings,
    path=persist_directory,
    collection_name="my_documents",
    force_recreate=True,
)

    # print("\n qdrant: ",qdrant )
    return qdrant







