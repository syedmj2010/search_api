from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
import  os
from dotenv import load_dotenv
load_dotenv()

# Load OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
#print(openai_api_key)
if not openai_api_key:
    raise ValueError("The environment variable OPENAI_API_KEY is not set.")

# Initialize the OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

# Example usage with the LLM
#response = llm("What is the capital eeeee  of France?")
#print(response)

loader=TextLoader('speech.txt', encoding='utf-8')

documents=loader.load()

#print(len(documents))
text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=80)
texts=text_splitter.split_documents(documents)
#print(len(texts))
#print(texts[0])

embeddings=OpenAIEmbeddings()
vector= embeddings.embed_query("Tesing the embedding model")
#print(len(vector))

doc_vector=embeddings.embed_documents([t.page_content for t in texts[:5]])
#print(len(doc_vector))
#print(doc_vector[0])




CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/vector_db"
COLLECTION_NAME = 'state_of_union_vectors'

db = PGVector.from_documents(
    embedding=embeddings,
    documents=texts,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

# Perform similarity search
query = "Tell about the summary of the speech file"
similar = db.similarity_search_with_score(query, k=2)

# Combine results and pass to LLM
context = "\n\n".join([doc[0].page_content for doc in similar])  # Extract only the content
print("Context from Similarity Search:\n")
print(context)

# Prepare the prompt
prompt = f"Based on the following context, answer the question:\n\n{context}\n\nQuestion: {query}"

# Get response from LLM
response = llm(prompt)
print("\nLLM Response:\n")
print(response.content)  # Print only the content of the LLM response
