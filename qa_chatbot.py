import os
import warnings
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
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

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_API_TOKEN")
os.environ["CHROMA_TELEMETRY_STATUS"] = "False"

if not HF_TOKEN:
    raise ValueError(
        "HF_API_TOKEN is missing! Please make sure you have restarted your "
        "Codespace container after adding the repository secret so it can load."
    )

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Cache built retrievers per uploaded file so we don't re-embed on every query.
# Keyed by absolute file path -> retriever object.
_RETRIEVER_CACHE = {}


## initialize LLM using HuggingFace
def get_llm():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    llm_endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",       # explicit: avoids ambiguous auto-detection
        max_new_tokens=1024,
        temperature=0.1,              # low temperature = more factual, less creative
        repetition_penalty=1.03,
        huggingfacehub_api_token=HF_TOKEN,
    )

    return ChatHuggingFace(llm=llm_endpoint)


## Document loader
def document_loader(file):
    loader = PyPDFLoader(file)
    loaded_document = loader.load()
    return loaded_document


## Text splitter
def text_splitter(data):
    splitter_obj = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter_obj.split_documents(data)
    return chunks


## Free embedding model running locally on your device
def llm_embedding():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


## Vector db
def vector_database(chunks):
    embedding_model = llm_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb


## Retriever (now cached per file path)
def retriever(file):
    if file in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE[file]

    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)

    # MMR retrieval pulls diverse chunks instead of near-duplicate top-k matches,
    # and k=6 gives the model enough surrounding context to answer correctly.
    retriever_obj = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20},
    )

    _RETRIEVER_CACHE[file] = retriever_obj
    return retriever_obj


# Join retrieved docs into a single text block, tagging each chunk with its
# source page so the model can cite where an answer came from.
def format_docs(docs):
    if not docs:
        return "No relevant context was found in the document."
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


## Modern QA Chain Implementation using LCEL
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)

    # Explicit grounding instructions: stops the model from inventing facts
    system_prompt = (
        "You are a careful assistant answering questions about a PDF document.\n"
        "Use ONLY the information in the context below to answer the question.\n"
        "If the answer is not contained in the context, say clearly: "
        "\"I don't know based on the provided document.\" Do not make anything up.\n"
        "When possible, mention the page number(s) your answer is based on.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # FIXED: Extract the raw string query from the input payload dictionary 
    # before passing it down into your vector database search engine step.
    rag_chain = (
        {
            "context": (lambda x: x["input"]) | retriever_obj | format_docs, 
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Invoke the chain structure with your key dictionary structure
    response_text = rag_chain.invoke({"input": query})
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
    title="Ai Chatbot",
    description="Upload a PDF document and ask any question. The chatbot uses a free Hugging Face model to answer, grounded strictly in the document content."
)

# Launch the app
if __name__ == "__main__":
    rag_application.launch(server_name="127.0.0.1", server_port=7860)