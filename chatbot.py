import os
from dotenv import load_dotenv
from langchain_community.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
import streamlit as st
from bs4 import BeautifulSoup

# Set the user-agent to make requests appear like they're coming from a real browser
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
headers = {
    'User-Agent': os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36")
}

# Streamlit page configuration
st.set_page_config(page_title="Java Code API Finder", layout="wide")
st.markdown("<h4 style='text-align: left;'>Java Code API Finder</h4>", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("The environment variable OPENAI_API_KEY is not set.")

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)

# Sitemap URL and Query Inputs
sitemap_url = st.text_input("Enter the sitemap URL:", "https://api.python.langchain.com/sitemap.xml", key="sitemap_url")
query = st.text_input(
    label="Test",
    placeholder="Search ",
    label_visibility="collapsed"
)
# defining a custom function to extract plain text from website
def get_plain_text_with_header(content: BeautifulSoup) -> str:
    # create clean text
    for items in content.find_all(class_="hsh-wrapper"):
      #collecting header and consequent text paragraph to improve the meaningfulness of the text
        data = " ".join(''.join([item.text for item in items.find_all(["h1","p"])]).split())
    return data

# Function to load documents using SitemapLoader
def load_documents_from_sitemap(sitemap_url):
    """Load all pages from the given sitemap URL using SitemapLoader."""
    try:
        # Use the user-provided sitemap URL and headers for the request
        loader = SitemapLoader(
            web_path="https://www.hs-harz.de/en/sitemap.xml",
            #filter_urls=["https://api.python.langchain.com/en/latest"],  
            parsing_function=get_plain_text_with_header,          
        )
        documents = loader.load()
        if not documents:
            st.warning("No documents found in the sitemap.")
            return None
        return documents
    except Exception as e:
        st.error(f"Error loading sitemap: {e}")
        return None

# Function Definitions
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

# Display the process button
st.button("Process Query")

# Process and display results when the query is entered
if query and sitemap_url:
    try:
        # Load documents from the sitemap
        documents = load_documents_from_sitemap(sitemap_url)
        if not documents:
            st.warning("No documents found. Please check the sitemap URL.")

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
            st.markdown(context)
            st.markdown("### Model Response:")
            st.markdown(f"```python\n{response.content}\n```")  # Display the generated response
        else:
            st.warning("No relevant code found.")
    except Exception as e:
        st.error(f"Error: {e}")
