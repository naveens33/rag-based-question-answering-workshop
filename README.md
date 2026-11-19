# Build Your Own AI Chatbot
## Designing and Implementing a RAG-Based Question Answering System

A beginner-friendly, code-first workshop demonstrating how to build a Retrieval-Augmented Generation (RAG) question-answering pipeline in Python.

**Workshop:** 20 November 2026  
**Environment:** Python, VS Code, Ollama, Qwen 2.5 3B, Sentence Transformers and FAISS

> **Important dataset notice:** `University_Academic_Handbook_2026.pdf` is a fictional training document created exclusively for this workshop. It is not an official policy document of New Horizon College of Engineering, BITS Pilani, or any real university. Do not use it for actual academic decisions.

## 1. What you will learn

You will compare three ways of answering a question:

1. **Standalone LLM:** Ask a local language model a question without supplying institutional information.
2. **LLM with manual context:** Include a relevant passage directly in the prompt.
3. **Automated RAG:** Extract information from a PDF, split it into chunks, convert chunks into embeddings, index them with FAISS, retrieve relevant passages, and provide those passages to the LLM.

By the end, you will have run a complete, local RAG pipeline from a Python script.

## 2. How the RAG pipeline works

```text
                 FICTIONAL HANDBOOK (PDF)
                            |
                            v
                     PDF TEXT EXTRACTION
                            |
                            v
                       TEXT CHUNKING
                            |
                            v
                    EMBEDDING GENERATION
                            |
                            v
                       FAISS INDEX
                            |
                            |
USER QUESTION --------------+
       |
       v
QUESTION EMBEDDING
       |
       v
RETRIEVE TOP-K RELEVANT CHUNKS
       |
       v
BUILD PROMPT WITH RETRIEVED CONTEXT
       |
       v
LOCAL LLM (OLLAMA + QWEN 2.5 3B)
       |
       v
GENERATED ANSWER
```

RAG retrieves relevant external information at question-answering time and supplies it to the language model. It does **not** train or fine-tune the model.

## 3. Project structure

```text
rag-based-question-answering-workshop/
|
|-- data/
|   |-- University_Academic_Handbook_2026.pdf
|
|-- rag_workshop.py
|-- requirements.txt
|-- README.md
```

## 4. Prerequisites

- Python 3.10 or later
- Visual Studio Code
- Ollama installed and running
- Qwen 2.5 3B model available in Ollama
- Internet access for the initial Python package and embedding-model downloads

### Check Ollama

```bash
ollama --version
```

Download the model if it is not already available:

```bash
ollama pull qwen2.5:3b
```

Confirm that the model is available:

```bash
ollama list
```

## 5. Set up the project in VS Code

Open the project folder in VS Code and create a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the Python executable inside `.venv` directly or follow your machine's approved PowerShell execution policy.

### Install dependencies

The `requirements.txt` file should contain:

```text
pymupdf
sentence-transformers
faiss-cpu
ollama
numpy
```

Install the packages:

```bash
python -m pip install -r requirements.txt
```

## 6. Run the workshop

Ensure the PDF is located at:

```text
data/University_Academic_Handbook_2026.pdf
```

Ensure Ollama is running and the `qwen2.5:3b` model is available.

Run:

```bash
python rag_workshop.py
```

The first run may take longer because the `all-MiniLM-L6-v2` embedding model must be downloaded. Later runs can reuse the locally cached model.

## 7. Explore the code sections

| Section | What it demonstrates |
|---|---|
| 0 | Standalone LLM without RAG |
| 0.1 | LLM with manually supplied context |
| 1 | Extract text from the PDF |
| 2 | Split extracted text into overlapping chunks |
| 3 | Convert chunks into numerical embeddings |
| 4 | Store embeddings in a FAISS vector index |
| 5 | Retrieve the most relevant chunks for a question |
| 6 | Supply retrieved context to the LLM and generate an answer |
| 7 | Run a small set of test questions |

### Try changing the questions

In Section 5, change the `question` variable and run the script again.

Example questions:

```python
question = "What is the minimum attendance requirement?"
question = "How is attendance calculated?"
question = "What percentage of marks comes from internal assessment?"
question = "What are the hostel accommodation fees?"
question = "Does medical leave automatically exempt a student from attendance?"
```

Observe the retrieved source pages and the generated answers.

## 8. Technologies used

- **PyMuPDF:** Extracts text from the PDF.
- **Sentence Transformers (`all-MiniLM-L6-v2`):** Converts text into numerical embeddings.
- **FAISS:** Searches the vector index for relevant chunks.
- **Ollama:** Runs the local language model.
- **Qwen 2.5 3B:** Generates answers using the retrieved context.
- **NumPy:** Handles embedding arrays.

## 9. Important concepts and limitations

- Chunk size and overlap influence what information is available to retrieval.
- Embeddings represent text as numerical vectors so semantically related text can be searched.
- FAISS `IndexFlatL2` uses L2 (Euclidean) distance. A lower distance means a closer vector in this search; it is not a probability or a guaranteed confidence score.
- Retrieval quality depends on the document, chunking strategy, embedding model and question.
- RAG does not guarantee factual answers. The model can still misunderstand context or add unsupported details.
- Check generated answers against the retrieved source passages.
- The five questions in this workshop are a small functional demonstration, not a statistically meaningful accuracy benchmark.

## 10. Dataset disclaimer

The included University Academic Handbook is fictional and intended only for educational demonstration. All institutions, policies, rules and values in it are fictional. It is not an official document of New Horizon College of Engineering, BITS Pilani, or any real institution.

## 11. Workshop outcome

You will have built and executed a basic PDF-based RAG question-answering pipeline locally, and seen the difference between a standalone LLM, manual context injection and automated retrieval.

---

**Learning by building:** Open `rag_workshop.py`, follow the numbered sections, run the code, and experiment with your own questions.
