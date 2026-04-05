# Custom prompt template
    prompt_template = """ Use the following context to answer the question. Be concise and factual.
    Context: {context}
    Question: {question}
    Answer:"""

    PROMPT = PromptTemplate(
        template = prompt_template,
        input_variables = ["context", "question"]
    )

    # From last line of chain
    # Use custom prompt template above
        chain_type_kwargs={"prompt": PROMPT}

# Alternative to use online version of Llama via API (instead of local)
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os

# Llama 3.2 1B via API
base_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    temperature=0.1,
    max_new_tokens=600,
)

llm_hub = ChatHuggingFace(llm=base_llm)

# Could also use faster versions on cloud
# Vision version can interpret images and charts
repo_id="meta-llama/Llama-3.2-3B-Instruct"  # Faster on cloud
repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct"  # Vision model

# output with original prompt format:
"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Ocmulgee Reptiles Jason Barron 457 Second St Rhine, GA 31077 United States ocmulgeereptiles@gmail.com Invoice #: 0026 Invoice Date: Dec 1, 2020 Due date: Dec 1, 2020 Amount due: $530.00 INVOICE Partially paid Bill To: Jessica Bean jessica.bean@bastyr.edu Ship To: Jessica Bean Description Quantity Price Amount BEL (Bamboo Butter) Ball Python 65-OR-Ball-20F-BamBut-001 1 $600.00 $600.00 Subtotal $600.00 Shipping $50.00 Total $650.00 Amount paid -$120.00 Amount due $530.00 USD Terms and Conditions Refunds: We cannot offer any cash refunds or credit card refunds, however an exchange of equal or lesser value can be arranged if notified within 24 hours. Buyer is responsible for all shipping cost. Scan. Pay. Go Question: What type of ball python was ordered, how much did it cost, and what was the date of the transaction? Helpful Answer: According to the invoice, the ball python was partially paid, and the total amount paid was $650"
"portion of your response be concise and factual. In that part of the response do NOT give additional information and do NOT repeat information. Context: Ocmulgee Reptiles Jason Barron 457 Second St Rhine, GA 31077 United States ocmulgeereptiles@gmail.com Invoice #: 0026 Invoice Date: Dec 1, 2020 Due date: Dec 1, 2020 Amount due: $530.00 INVOICE Partially paid Bill To: Jessica Bean jessica.bean@bastyr.edu Ship To: Jessica Bean Description Quantity Price Amount BEL (Bamboo Butter) Ball Python 65-OR-Ball-20F-BamBut-001 1 $600.00 $600.00 Subtotal $600.00 Shipping $50.00 Total $650.00 Amount paid -$120.00 Amount due $530.00 USD Terms and Conditions Refunds: We cannot offer any cash refunds or credit card refunds, however an exchange of equal or lesser value can be arranged if notified within 24 hours. Buyer is responsible for all shipping cost. Scan. Pay. Go Question: what is the invoice number? Answer: 0026 Invoice Date: Dec 1, 2020 Due Date:"

# Possible update to document processor function to delete pdfs after processing:
import os

def process_document(document_path):
    global conversation_retrieval_chain
    
    try:
        # Reset chain
        conversation_retrieval_chain = None
        
        logger.info(f"Loading document from path: {document_path}")
        
        # Load PDF
        loader = PyPDFLoader(document_path)
        documents = loader.load()
        
        # Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=64
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        db = Chroma.from_documents(texts, embedding=embeddings)
        
        # Create RAG chain
        conversation_retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm_hub,
            chain_type="stuff",
            retriever=db.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 3, 'lambda_mult': 0.25}
            ),
            return_source_documents=False,
            input_key="question",
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("RAG chain created successfully.")
        
    finally:
        # Delete file after processing (success or failure)
        if os.path.exists(document_path):
            os.remove(document_path)
            logger.info(f"Deleted uploaded file: {document_path}")