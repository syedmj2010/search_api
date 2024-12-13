import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain.agents import initialize_agent, Tool, AgentType
import streamlit as st

# Streamlit page configuration
st.set_page_config(page_title="Java Code API Finder", layout="wide")
# Streamlit page configuration
#st.title("Java Code API Finder")
st.markdown("<h4 style='text-align: left;'>Java Code API Finder</h4>", unsafe_allow_html=True)


# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("The environment variable OPENAI_API_KEY is not set.")

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

# Folder Path and Query Inputs
#folder_path = st.text_input("Enter the folder path containing Java files:", r'C:\Workspace\eazyBytes\spring-security', key="folder_path")
folder_path = r'C:\Workspace\-----'
#query = st.text_input("Enter your query related to the Java code:", "Can you provide the api for user creation?", key="query")
query = st.text_input(  
    label="",   
    placeholder="Search for a Java API or Java code snippet..."
)

# Function Definitions
def load_documents(folder_path):
    """Load all Java files from the given folder."""
    file_paths = glob.glob(f"{folder_path}\\**\\*.java", recursive=True)
    if not file_paths:
        st.warning("No Java files found in the provided folder path.")
        return None
    documents = []
    for file_path in file_paths:
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
    return documents

def split_texts(documents):
    """Split documents into text chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
    texts = text_splitter.split_documents(documents)
    st.write(f"Number of text chunks: {len(texts)}")
    return texts

def get_embeddings_and_search(query, texts):
    """Generate embeddings and perform similarity search."""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/vector_db"
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=texts,
        collection_name='java_code_vectors',
        connection_string=CONNECTION_STRING,
    )
    return db.similarity_search_with_score(query, k=5)

def format_java_code(similar_results):
    """Format Java code and return it."""
    context = ""
    if isinstance(similar_results, list):
        seen_context = set()  # Set to keep track of seen snippets to avoid duplicates
        for doc, score in similar_results:
            if hasattr(doc, 'page_content') and doc.page_content not in seen_context:
                seen_context.add(doc.page_content)
                context += doc.page_content + "\n\n"
        if context:
            formatted_code = f"```java\n{context}\n```"
            return formatted_code, context
    return "No relevant code found.", ""

def generate_response(query, context):
    """Generate response based on the given query and context."""
    prompt = f"Based on the following context, answer the question:\n\n{context}\n\nQuestion: {query}"
    return llm.invoke(prompt)

# Initialize tools for the agent
tools = [
    Tool(
        name="Document Loader",
        func=load_documents,
        description="Load all Java files from the given folder."
    ),
    Tool(
        name="Text Splitter",
        func=split_texts,
        description="Split documents into text chunks."
    ),
    Tool(
        name="Embedding Search",
        func=get_embeddings_and_search,
        description="Generate embeddings and perform a similarity search."
    ),
    Tool(
        name="Format Java Code",
        func=format_java_code,
        description="Format the returned Java code snippet."
    ),
    Tool(
        name="Generate Model Response",
        func=generate_response,
        description="Generate a response based on the given query and context."
    ),
]

# Initialize the agent
agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


# Process and display results when the query is entered
if query:
    if st.button("Process Query"):
        try:
            # Load documents from the folder
            documents = load_documents(folder_path)
            # if documents is None:  # Stop further processing if no documents are found
            #     st.warning("No Java files found in the provided folder path. Please try again with a valid folder.")
            #     return  # Removing return causes further processing to stop

            # Split the documents into chunks
            texts = split_texts(documents)

            # Perform similarity search with the query
            similar_results = get_embeddings_and_search(query, texts)

            # Format Java code and get the context
            formatted_code, context = format_java_code(similar_results)

            # Now pass the context to the generate_response function
            if context:
                response = generate_response(query, context)
                st.markdown("### Found Code Snippets:")
                st.markdown(formatted_code)  # Display formatted Java code
                st.markdown("### Model Response:")                
                #st.write(response.content)  # Display model's response
                #st.markdown(f"```markdown\n{response.content}\n```")
                #st.markdown(f"### Model Response\n\n{response.content}")
                st.markdown(f"```python\n{response.content}\n```")              
            else:
                st.warning("No relevant code found.")

        except Exception as e:
            st.error(f"Error: {e}")

