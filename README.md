# RAG PDF Chatbot - Open Source Rebuild

Retrieval-Augmented Generation (RAG) chatbot that processes PDF documents and answers questions using open-source models, rebuilt from a proprietary platform-dependent implementation.

## Project Overview

This project represents a major architectural refactoring of an IBM AI Developer course RAG chatbot, transitioning from proprietary API-based models to a fully open-source, locally-runnable implementation using Llama 3.2 1B-Instruct.

## Key Engineering Achievements

### 1. Model Migration: Proprietary → Open Source

**Challenge:** Original implementation relied on Llama 3.3 70B via IBM WatsonX API, creating vendor lock-in and requiring internet connectivity and API costs.

**Solution:** Migrated to **meta-llama/Llama-3.2-1B-Instruct**, an open-source model that runs entirely locally.

**Technical Benefits:**
- ✅ No API dependencies or costs
- ✅ Full data privacy (all processing local)
- ✅ Works offline
- ✅ Faster inference (1B vs 70B model)
- ✅ Reproducible and auditable

**Trade-offs Evaluated:**
- Model quality: 1B model sufficient for document Q&A tasks
- Speed vs accuracy: Chose speed for better UX in PDF querying use case
- Resource requirements: 1B model runs on consumer hardware vs 70B requiring enterprise infrastructure

---

### 2. Dependency Resolution & Version Conflicts

**Challenge:** Switching from WatsonX API to HuggingFace transformers pipeline introduced complex dependency conflicts between `transformers`, `langchain`, and `langchain-huggingface` packages.

**Root Cause Analysis:**
- `langchain-huggingface` package incompatible with `transformers` 5.x
- Newer LangChain versions deprecated certain import paths
- Multiple packages claiming ownership of same interface classes

**Solution:**
```python
# Final working configuration
transformers==4.47.1  # Pinned to avoid langchain-huggingface conflicts
langchain-community   # Used instead of langchain-huggingface
```

**Updated imports:**
```python
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA  
```

**Process:** Systematic testing of version combinations, analyzing error traces, and consulting package changelogs to identify compatible versions.

---

### 3. Model Configuration & Warning Suppression

**Challenge:** Pipeline initialization generated cryptic warnings about conflicting generation configs and missing token IDs.

**Approach:** Rather than suppressing warnings, investigated root causes by:
- Examined warnings and recommendations, checking documentation as needed
- Tested parameter combinations

**Solutions Implemented:**
```python
model_kwargs = {
    "generation_config": None,  # Prevent conflict with model's built-in config
    "pad_token_id": 128001,     # Previously warned that this was not set
    # ... other params
}
```

**Result:** Clean execution with no warnings, proper understanding of model behavior.

---

### 4. Chain State Management Bug Fix

**Issue Discovered:** Processing multiple PDFs in sequence caused response contamination - answers from previous documents leaked into new queries.

**Root Cause:** `conversation_retrieval_chain` object persisted between document uploads, retaining old vector store references.

**Fix:**
```python
def process_document(document_path):
    global conversation_retrieval_chain

    # ... PDF processing
    
    conversation_retrieval_chain = None  # Explicit reset before rebuild
    
    # ... new chain creation
```

**Impact:** Eliminated cross-document contamination, ensured clean state per upload.

---

### 5. Response Truncation Debugging

**Problem:** Model responses consistently cut off mid-sentence regardless of content length.

**Debugging Process:**

**Hypothesis 1:** `max_new_tokens` in model kwargs is too low
- Tested: Increased from 512 → 1024 → None
- Result: No effect ❌

**Hypothesis 2:** EOS token triggering early
- Tested: `eos_token_id=None`
- Result: No effect ❌

**Hypothesis 3:** Minimum token constraint issue
- Tested: `min_new_tokens=100`, then 300
- Result: No effect ❌

**Hypothesis 4:** Pipeline-level limit overriding model kwargs
- **Root cause identified:** Pipeline initialization had default `max_length` parameter
- **Fix:**
```python
# WRONG - default pipeline max_length overrides everything
llm_pipeline = pipeline(task="text-generation", model=model_name)

# CORRECT - set high limit and let model_kwargs control generation
llm_pipeline = pipeline(task="text-generation", model=model_name, max_new_tokens=1024, model_kwargs={},)
```
- Result: Full responses generated ✅

**Key Learning:** Parameter precedence matters - pipeline-level configs override model-level configs in transformers library.

---

### 6. Output Formatting & Post-Processing

**Challenge:** Raw model output included reasoning traces and template artifacts:
```
Context: [long context]
Question: What is the invoice number?
Answer: Invoice #0026
```

**Solution:** Implemented string parsing to extract clean answers:
```python
# Extract content after "Answer:" marker
answer = output["result"]
answer_index = answer.find("Answer: ") + 8
answer = answer[answer_index:]
```

**Result:** User sees only `"Invoice #0026"` - clean, professional output.

---

### 7. Answer Quality & Consistency Improvements

**Problem Identified:** 
- Same question yielded different answers across queries
- Model sometimes copied context verbatim instead of extracting information
- Irrelevant information included in responses

**Example Issue:**
```
Q: What is the invoice number?
A1: Invoice #0026 ✅
A2: 0026 Invoice Date: Dec 1, 2020 Due Date: ❌
```

**Root Cause Analysis:**
1. **Temperature too low (0.1):** Model behaving deterministically, pattern-matching rather than reasoning
2. **Prompt insufficiently specific:** No clear instruction to be concise
3. **Context bleeding:** Model regurgitating surrounding text

**Solutions Applied:**

**A. Temperature Optimization:**
```python
"temperature": 0.3,  # Increased from 0.1
```
- 0.1: Too deterministic, copies context
- 0.3: Allows reasoning while maintaining factual accuracy
- Trade-off: Slight variation acceptable for better understanding

**B. Prompt Engineering:**
```python
prompt_template = """You are a helpful assistant answering questions about documents. Use the following context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Instructions: Answer ONLY the specific question asked. Be concise. Do not include extra information from the context unless directly relevant to answering the question.

Answer:"""
```

**Impact:** 
- Consistent factual extraction
- Minimal extraneous information
- Natural language responses

---

### 8. Context Window Optimization

**Challenge:** Llama 3.2 1B has 8,192 token context limit. Large PDFs with high `k` (retrieval chunks) were filling the window, leaving insufficient space for responses.

**Analysis:**
```
Context chunks (k=6): ~4,000 tokens
Prompt template: ~200 tokens
Question: ~50 tokens
Remaining for answer: ~3,942 tokens
```

**Solution:**
```python
retriever=db.as_retriever(
    search_type="mmr", # for balance of similarity and diversity in chunks
    search_kwargs={'k': 3, 'lambda_mult': 0.25}  # Reduced from k=6
)
```

**Rationale:**
- MMR (Maximal Marginal Relevance) already optimizes for diversity
- 3 high-quality chunks > 6 potentially redundant chunks
- More token budget for comprehensive answers

**Result:** Better response quality, no context overflow issues.

---

## Architecture

```
PDF Upload
    ↓
PyPDFLoader (extract text)
    ↓
RecursiveCharacterTextSplitter (chunk_size=1024, overlap=64)
    ↓
HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)
    ↓
ChromaDB (vector storage)
    ↓
MMR Retrieval (k=3, diversity-optimized)
    ↓
RetrievalQA Chain (stuff method)
    ↓
Llama 3.2 1B-Instruct (local inference)
    ↓
Post-processed Answer
```

---

## Technical Stack

**Core Dependencies:**
- `transformers==4.47.1` - Model inference
- `langchain` - RAG orchestration
- `langchain-community` - HuggingFace integration
- `chromadb` - Vector storage
- `sentence-transformers` - Embeddings
- `pypdf` - PDF parsing

**Model:**
- **LLM:** meta-llama/Llama-3.2-1B-Instruct
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2

---

## Key Configuration Parameters

```python
# LLM Generation
max_new_tokens: 512      # Response length limit
temperature: 0.3         # Balanced creativity/factuality
top_p: 0.9               # Nucleus sampling
do_sample: True          # Enable sampling (non-greedy)

# Text Splitting
chunk_size: 1024         # Characters per chunk
chunk_overlap: 64        # Overlap for context continuity

# Retrieval
search_type: "mmr"       # Maximal Marginal Relevance
k: 3                     # Number of chunks retrieved
lambda_mult: 0.25        # Diversity vs relevance balance

# Chain
chain_type: "stuff"      # Combine all chunks in single prompt
```

---

## Future Enhancements

**Identified but not yet implemented:**

1. **Automatic PDF cleanup:** Delete uploaded PDFs after processing to prevent disk bloat

2. **Multi-document persistence:** Support querying across multiple PDFs simultaneously

3. **Model upgrading path:** Document migration to Llama 3.2 3B for quality improvement while maintaining local deployment

---

## Engineering Principles Demonstrated

1. **Incremental debugging:** Systematic hypothesis testing rather than random changes
2. **Root cause analysis:** Understanding *why* before implementing fixes
3. **Documentation-driven development:** Consulting model cards and library docs before Stack Overflow
4. **Trade-off evaluation:** Explicit analysis of speed/quality/cost decisions
5. **Clean state management:** Proper reset logic to prevent subtle bugs
6. **Parameter tuning:** Data-driven optimization of model and retrieval settings

---

## Setup & Usage

**Requirements:**
- Python 3.10+
- 8GB+ RAM (for model inference)
- HuggingFace account (for Llama 3.2 access)

**Installation:**
```bash
pip install -r requirements.txt
huggingface-cli login  # Provide token
```

**Accept Llama 3.2 license:**
Visit https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct and accept terms

**Run:**
```bash
python server.py
```

---

## License

Model usage subject to Llama 3.2 Community License Agreement. 
Application code is licensed under the Apache License 2.0.

Original course template © IBM Corporation. 
Significant architectural modifications © 2026 Jessica Bean.

See [LICENSE](LICENSE) file for details.

---

### Attributions

Original course template © IBM Corporation, licensed under Apache 2.0.
Significant modifications and architectural improvements by Jessica Bean, 2026.

---

## Contact

**Jessica Bean**
- GitHub: [@jessicabean](https://github.com/jessicabean)
- LinkedIn: [linkedin.com/in/jessica-bean-ms](https://www.linkedin.com/in/jessica-bean-ms/)
