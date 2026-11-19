# ============================================================
# BUILD YOUR OWN AI CHATBOT
# RAG-Based Question Answering System
#
# Fictional University Academic Handbook 2026
# This handbook is a teaching example. It is NOT the policy
# of New Horizon College or any other real institution.
# ============================================================


# ============================================================
# SECTION 0: STANDALONE LLM (WITHOUT RAG)
# ============================================================

import ollama

MODEL_NAME = "qwen2.5:3b"

question = "What is the minimum attendance requirement at Fictional University?"

response = ollama.chat(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": question}
    ]
)

print("\n" + "=" * 65)
print("SECTION 0: STANDALONE LLM")
print("=" * 65)
print("Question:", question)
print("\nAnswer:")
print(response["message"]["content"])


# ============================================================
# SECTION 0.1: LLM WITH MANUALLY SUPPLIED CONTEXT
# ============================================================

# This passage was manually copied from the fictional handbook.
manual_context = """
A student must maintain at least 80% attendance
in each registered course to be eligible to appear
for the end-semester examination in that course.
"""

prompt = f"""
Answer the question using the information provided below.

Context:
{manual_context}

Question:
{question}

Answer:
"""

response = ollama.chat(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\n" + "=" * 65)
print("SECTION 0.1: LLM WITH MANUAL CONTEXT")
print("=" * 65)
print("Manually supplied context:")
print(manual_context)
print("Answer:")
print(response["message"]["content"])


# ============================================================
# SECTION 1: EXTRACT TEXT FROM THE PDF
# ============================================================

import pymupdf

pdf_path = "data/University_Academic_Handbook_2026.pdf"

document = pymupdf.open(pdf_path)

pages = []

for page_number, page in enumerate(document):
    page_text = page.get_text()

    pages.append({
        "page": page_number + 1,
        "text": page_text
    })

document.close()

print("\n" + "=" * 65)
print("SECTION 1: PDF TEXT EXTRACTION")
print("=" * 65)
print("Total pages extracted:", len(pages))
print("\nText from the first page:")
print(pages[0]["text"])


# ============================================================
# SECTION 2: SPLIT THE TEXT INTO CHUNKS
# ============================================================

# The PDF's extracted text is divided into chunks of approximately 900 characters each.
chunk_size = 900
# The next chunk repeats the last 150 characters from the previous chunk to preserve context between consecutive chunks.
# Why do we need overlap?
# Suppose an important sentence starts at the end of Chunk 1 and continues into Chunk 2. Without overlap, the meaning could get separated.
chunk_overlap = 150

chunks = []

for page in pages:
    text = page["text"]
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        # Keep the original PDF page number with each chunk.
        chunks.append({
            "page": page["page"],
            "text": chunk_text
        })

        # Move forward while retaining some overlapping text.
        start += chunk_size - chunk_overlap

print("\n" + "=" * 65)
print("SECTION 2: TEXT CHUNKING")
print("=" * 65)
print("Chunk size:", chunk_size, "characters")
print("Chunk overlap:", chunk_overlap, "characters")
print("Total chunks created:", len(chunks))
print("\nFirst chunk:")
print(chunks[0]["text"])
print("Source page:", chunks[0]["page"])


# ============================================================
# SECTION 3: CONVERT CHUNKS INTO EMBEDDINGS
# ============================================================

from sentence_transformers import SentenceTransformer

# This model converts text into 384-dimensional vectors.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_texts = [chunk["text"] for chunk in chunks]

embeddings = embedding_model.encode(
    chunk_texts,
    convert_to_numpy=True
)

print("\n" + "=" * 65)
print("SECTION 3: EMBEDDING GENERATION")
print("=" * 65)
print("Number of embeddings:", len(embeddings))
print("Embedding shape:", embeddings.shape)
print("Embedding dimensions:", embeddings.shape[1])


# ============================================================
# SECTION 4: CREATE THE FAISS VECTOR INDEX
# ============================================================

import faiss
import numpy as np

# FAISS expects float32 vectors for this index.
embeddings = embeddings.astype("float32")

dimension = embeddings.shape[1]

# IndexFlatL2 performs an exact search using L2 (Euclidean) distance.
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("\n" + "=" * 65)
print("SECTION 4: FAISS VECTOR INDEX")
print("=" * 65)
print("FAISS index created successfully!")
print("Total vectors stored:", index.ntotal)
print("Vector dimensions:", index.d)


# ============================================================
# SECTION 5: RETRIEVE RELEVANT CHUNKS
# ============================================================

# Change this question to experiment with retrieval.
question = "What are the hostel accommodation fees?"
# question = "What is the minimum attendance requirement?"

top_k = 3

question_embedding = embedding_model.encode(
    [question],
    convert_to_numpy=True
).astype("float32")

distances, indices = index.search(question_embedding, k=top_k)

print("\n" + "=" * 65)
print("SECTION 5: RETRIEVAL")
print("=" * 65)
print("Question:", question)
print("\nRetrieved passages:")

for rank, chunk_index in enumerate(indices[0], start=1):
    retrieved_chunk = chunks[chunk_index]

    print(f"\nResult {rank}")
    print("Source page:", retrieved_chunk["page"])
    print("L2 distance:", round(float(distances[0][rank - 1]), 4))
    print("Text:")
    print(retrieved_chunk["text"])
    print("-" * 65)


# ============================================================
# SECTION 6: GENERATE AN ANSWER USING RAG + OLLAMA
# ============================================================

# Combine the retrieved chunks into one context.
retrieved_context = ""

for chunk_index in indices[0]:
    retrieved_context += chunks[chunk_index]["text"] + "\n\n"

# Show the evidence that will be passed to the language model.
source_pages = sorted({
    chunks[chunk_index]["page"]
    for chunk_index in indices[0]
})

prompt = f"""
Answer the question using only the provided context.

If the answer is not available in the context,
say that the handbook does not provide this information.

Do not add details that are not supported by the context.

Context:
{retrieved_context}

Question:
{question}

Answer:
"""

print("\n" + "=" * 65)
print("SECTION 6: RAG ANSWER GENERATION")
print("=" * 65)
print("Retrieved source pages:", source_pages)
print("\nContext supplied to the LLM:")
print(retrieved_context)

response = ollama.chat(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nGenerated answer:")
print(response["message"]["content"])


# ============================================================
# SECTION 7: SIMPLE RAG EVALUATION
# ============================================================

import time

test_questions = [
    "What is the minimum attendance requirement?",
    "How is attendance calculated?",
    "What percentage of marks comes from internal assessment?",
    "What are the hostel accommodation fees?",
    "Does medical leave automatically exempt a student from attendance?"
]

print("\n" + "=" * 65)
print("SECTION 7: SIMPLE RAG EVALUATION")
print("=" * 65)

for test_number, test_question in enumerate(test_questions, start=1):

    # 1. Convert the question into an embedding.
    question_embedding = embedding_model.encode(
        [test_question],
        convert_to_numpy=True
    ).astype("float32")

    # 2. Retrieve the three closest chunks from FAISS.
    distances, indices = index.search(question_embedding, k=top_k)

    # 3. Build the context from the retrieved chunks.
    context = ""

    for chunk_index in indices[0]:
        context += chunks[chunk_index]["text"] + "\n\n"

    # 4. Keep track of the source pages for verification.
    source_pages = sorted({
        chunks[chunk_index]["page"]
        for chunk_index in indices[0]
    })

    # 5. Construct the prompt for the language model.
    evaluation_prompt = f"""
Answer using only the provided context.

If the answer is unavailable, say:
The handbook does not provide this information.

Do not add details that are not supported by the context.

Context:
{context}

Question:
{test_question}

Answer:
"""

    # 6. Generate the answer and measure response time.
    start_time = time.time()

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": evaluation_prompt}
        ]
    )

    elapsed_time = time.time() - start_time

    print(f"\nTest {test_number}")
    print("QUESTION:", test_question)
    print("SOURCE PAGES:", source_pages)
    print("ANSWER:", response["message"]["content"])
    print("RESPONSE TIME:", round(elapsed_time, 2), "seconds")
    print("-" * 65)


# ============================================================
# WORKSHOP TAKEAWAY
# ============================================================
# RAG retrieves relevant information from an external source
# and supplies it to the LLM at question-answering time.
#
# RAG does not guarantee that every generated statement is
# correct. Always compare the answer with its retrieved source.
# ============================================================
