import os
import warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
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
os.environ["CHROMA_TELEMETRY_STATUS"] = "False"

# No API token needed anymore: the LLM runs locally through Ollama, and the
# embedding model runs locally too (sentence-transformers downloads once,
# then runs on-device with no network calls or rate limits).

# Model to use. Requires Ollama installed and running locally
# (https://ollama.com), plus the model pulled once via:
#   ollama pull llama3.1:8b
# If your machine is CPU-only and 8B is too slow, swap in a lighter model:
#   ollama pull qwen2.5:3b     or     ollama pull phi3:mini
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Cache built retrievers per uploaded file so we don't re-embed on every query.
# Keyed by absolute file path -> retriever object.
_RETRIEVER_CACHE = {}


## initialize LLM using local Ollama server
def get_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.1,   # low temperature = more factual, less creative
        num_predict=1024,  # equivalent to max_new_tokens
    )


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

    # Explicit grounding instructions: this is what stops the model from
    # guessing or inventing facts not present in the retrieved context.
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

    # Note: invoking with {"input": query} means each branch of this parallel
    # map receives that whole dict, not a bare string — so the "context"
    # branch must pull the raw query string back out before handing it to
    # the retriever (which expects a string, not a dict).
    rag_chain = (
        {
            "context": (lambda x: x["input"]) | retriever_obj | format_docs,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

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
    description="Upload a PDF document and ask any question. The chatbot runs fully locally via Ollama, grounded strictly in the document content."
)

# Launch the app
if __name__ == "__main__":
    rag_application.launch(server_name="127.0.0.1", server_port=7860)