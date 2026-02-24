import os
import torch
import logging
import flask

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

def init_llm():
    '''
    Initialize Llama 3.21B locally, wrap for LangChain, set embeddings
    '''
    # Specify global variables modified within the function
    global llm_hub, embeddings
    
    logger.info("Loading Llama 3.2 chatbot - this may take a moment...")

    # Llama 3.21B - open source 2024 chatbot
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    # Initialize llm_pipeline using pipeline
    llm_pipeline = pipeline(
        task="text-generation", 
        model=model_name,
        max_new_tokens=1024,
        model_kwargs={},
    )

    # Wrap llm_pipeline for LangChain to create compatible object
    llm_hub = HuggingFacePipeline(
        pipeline = llm_pipeline,
        model_kwargs = {
            # Equates to roughly ~400 word response
            "max_new_tokens": 512,
            # Temperature from 0 (fully deterministic) to 2 (very high randomness)
            # For RAG should be highly deterministic, but for non-technical fields 0.2-0.4 will give more natural language (Claude)
            "temperature": 0.1,
            # Nucleus sampling: What total percent of next word probabilities to sum before exclude possibilities
            # 0.5 would be maximally factual, 1.0 would include all, Claude recommends 0.9
            "top_p": 0.6,
            # True allows for some natural language variation and rephrase
            # False very robotic: means always picking the most likely conclusion (low variation)
            "do_sample": True,
            # Ignore model's internal generation_config settings (otherwise get informational warnings about conflicts)
            "generation_config": None,
            # Set pad_token_id explicitly to eliminate informational warnings about setting it
            "pad_token_id": 128001,
            #"streaming": True,
            #"eos_token_id": [128009],
        }
    )

    logger.debug("Llama 3.21B initialized: %s", llm_hub)

    # Initialize embeddings compatible with langchain
    # Act as search engine within pdf to work with LLM
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}
    )

    logger.debug("Embeddings initialized with model device: %s", DEVICE)

def process_document(document_path):
    '''
    Load a pdf from document_path, split into chunks, create embeddings database, and set conversation_retrieval_chain
    '''
    
    global conversation_retrieval_chain

    logger.info("Loading document from path: %s", document_path)
    # Load the document
    loader =  PyPDFLoader(document_path)
    documents = loader.load()
    logger.debug("Loaded %d document(s)", len(documents))

    # Split the document into chunks, set chunk_size=1024, and chunk_overlap=64. assign it to variable text_splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 64)
    texts = text_splitter.split_documents(documents)
    logger.debug("Document split into %d text chunks", len(texts))

    # Create an embeddings database using Chroma from the split text chunks.
    logger.info("Initializing Chroma vector store from documents...")
    db = Chroma.from_documents(texts, embedding=embeddings)
    logger.debug("Chroma vector store initialized.")

    # Custom prompt template
    prompt_template = """You are a helpful assistant answering questions about documents. Use the following context to answer the question.If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Instructions: Answer ONLY the specific question asked. Be concise. Do not include extra information from the context unless directly relevant to answering the question.
    Answer:"""

    PROMPT = PromptTemplate(
        template = prompt_template,
        input_variables = ["context", "question"]
    )

    # Optional: Log available collections if accessible (this may be internal API)
    try:
        collections = db._client.list_collections()  # _client is internal; adjust if needed
        logger.debug("Available collections in Chroma: %s", collections)
    except Exception as e:
        logger.warning("Could not retrieve collections from Chroma: %s", e)

    # Reset conversation_retrieval_chain to allow for a new document to be processed
    conversation_retrieval_chain = None

    # Build the QA chain, which utilizes the LLM and retriever for answering questions. 
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        # Reads retrieved contenxt & prompt and generates final answer
        llm=llm_hub,
        # Put all chunks in one prompt
        chain_type="stuff", 
        # Search engine for the chunks (mmr)
        # Search parameters (return 6 chunks, prefer diversity - higher = more similar chunks)
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        # If True would also return source chunks (not just LLM answer text)
        return_source_documents=False,
        # Expects a question format in prompt
        input_key="question",
        # Use custom prompt template above
        chain_type_kwargs={"prompt": PROMPT},
    )
    logger.info("RetrievalQA chain created successfully.")

def clear_document():
    '''
    Upon page reload, clear conversation_retrieval_chain to allow a new document to be uploaded
        and prevent intermixing of document data.
    '''
    # Reset conversation_retrieval_chain to allow for a new document to be processed
    conversation_retrieval_chain = None

def process_prompt(prompt):
    '''
    Use LLama model to create a response using the prompt, conversation_retrieval_chain, and chat_history
    '''
    
    global conversation_retrieval_chain, chat_history
    
    # Check to confirm that chain exists
    if conversation_retrieval_chain is None:
        return "Please upload PDF first."

    logger.info("Processing prompt: %s", prompt)

    # Query the model using the .invoke() method
    output = conversation_retrieval_chain.invoke({"question": prompt, "chat_history": chat_history})
    answer = output["result"]
    answer = output["result"]
    answer_index = answer.find("Answer: ") + 8
    answer = answer[answer_index:]
    logger.debug("Model response: %s", answer)

    # Add user message to history
    chat_history.append((prompt, answer))

    logger.debug("Chat history updated. Total exchanges: %s", len(chat_history))

    # Return the model's response
    return answer

# Initialize the language model
init_llm()
logger.info("LLM and embeddings initialization complete.")
