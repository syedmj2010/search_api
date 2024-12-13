from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
import os
from dotenv import load_dotenv
import glob
from langchain_community.document_loaders import TextLoader

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("The environment variable OPENAI_API_KEY is not set.")

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

folder_path = r'C:\Workspace\LLM\svn_search\knowledge-base'

# Use glob to recursively find all .txt files
file_paths = glob.glob(f"{folder_path}\\**\\*.md", recursive=True)

# Load documents
documents = []
for file_path in file_paths:
    loader = TextLoader(file_path, encoding='utf-8')
    documents.extend(loader.load())

print(f"Number of documents loaded: {len(documents)}")

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
texts = text_splitter.split_documents(documents)
print(f"Number of text chunks: {len(texts)}")

# Generate embeddings
embeddings = OpenAIEmbeddings()

# Connect to PGVector database
CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/vector_db"
COLLECTION_NAME = 'state_of_union_vectors'

db = PGVector.from_documents(
    embedding=embeddings,
    documents=texts,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

# Perform similarity search
query = "Can you tell about Insurellm"
similar = db.similarity_search_with_score(query, k=2)

# Combine results and pass to LLM
context = "\n\n".join([doc[0].page_content for doc in similar])
prompt = f"Based on the following context, answer the question:\n\n{context}\n\nQuestion: {query}"

# Get response from LLM
response = llm.invoke(prompt)
print("LLM Response:", response.content)
