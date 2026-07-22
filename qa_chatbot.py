import os
import warnings

# Modern LangChain partner package for Hugging Face integration
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import gradio as gr

# Suppress warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

os.environ["CHROMA_TELEMETRY_STATUS"] = "False"

# --- CONFIGURATION ---
# Read the secret you added to your GitHub Codespace
HF_TOKEN = os.getenv("HF_API_TOKEN")

# Safety check to avoid NoneType errors and provide clear instructions
if not HF_TOKEN:
    raise ValueError(
        "HF_API_TOKEN is missing! Please make sure you have restarted your "
        "Codespace container after adding the repository secret so it can load."
    )

# Map your secret to the exact variable LangChain searches for internally
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN


## Free LLM via Hugging Face Inference API
def get_llm():
    # Using Mistral-7B-Instruct-v0.2 which is fast, high quality, and free over the API
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    watsonx_llm = HuggingFaceEndpoint(
        repo_id=model_id,
        max_new_tokens=256,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN
    )
    return watsonx_llm


## Document loader
def document_loader(file):
    loader = PyPDFLoader(file)
    loaded_document = loader.load()
    return loaded_document

## Text splitter
def text_splitter(data):
    # Fixed typo: renamed instantiation variable to avoid overwriting the class name
    splitter_obj = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
    )
    chunks = splitter_obj.split_documents(data)
    return chunks


## Free Embedding Model running locally on your device
def watsonx_embedding():
    # Uses sentence-transformers/all-MiniLM-L6-v2 which runs extremely fast on CPU
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

## Vector db
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

## Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever_obj = vectordb.as_retriever()
    return retriever_obj


# Helper function to join retrieved documents into a single text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


## Modern QA Chain Implementation using LCEL
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    
    # Setup explicit template to pass the document context into the model
    system_prompt = (
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Build the modern RAG chain using the pipe operator (|)
    rag_chain = (
        {"context": retriever_obj | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Invoke the chain structure directly
    response_text = rag_chain.invoke(query)
    return response_text


# Create Gradio interface using modern parameters
rag_application = gr.Interface(
    fn=retriever_qa,
    flagging_mode="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="Ai Chatbot (Hugging Face Free Tier)",
    description="Upload a PDF document and ask any question. The chatbot uses a free Hugging Face model to answer."
)

# Launch the app
if __name__ == "__main__":
    rag_application.launch(server_name="127.0.0.1", server_port=7860)
