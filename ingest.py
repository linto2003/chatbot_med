from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

DATA_PATH_PDF = 'data/pdf'
DATA_PATH_CSV = 'data/csv'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database
def create_vector_db():
    pdf_loader = DirectoryLoader(DATA_PATH_PDF, glob='*.pdf', loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()

    try:
        # Load CSV documents
        csv_loader = DirectoryLoader(DATA_PATH_CSV, glob='**/*.csv', loader_cls=CSVLoader)
        csv_documents = csv_loader.load()

    except Exception as e:
        print(f"Error loading documents: {e}")


    documents = pdf_documents 
    data = csv_documents

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    pdf_db = FAISS.from_documents(texts, embeddings)
    csv_db = FAISS.from_documents(data, embeddings)

    # Concatenate the two FAISS indexes
    db = FAISS.concat([pdf_db, csv_db])

  
    csv_db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()

