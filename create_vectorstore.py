from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load text file
loader = TextLoader("data/College_RAG_Knowledge_Base.txt", encoding="utf-8")

documents = loader.load()

print("Text file loaded successfully")
print("Number of documents:", len(documents))

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=30)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector database
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# Save vector database
vectorstore.save_local("vectorstore")

print("Vector database created successfully!")
print("Vector database created successfully!")