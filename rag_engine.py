"""RAG functions adapted from the existing Streamlit Python course assistant."""
import os
import re
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import joblib
import numpy as np
import requests
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "embeddings.joblib")
URLS_PATH = os.path.join(os.path.dirname(__file__), "urls.txt")
JINA_API_KEY = os.getenv("JINA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@lru_cache(maxsize=1)
def load_course_data():
    """Load existing course chunks and cache the document vectors once."""
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError("embeddings.joblib was not found in the Flask project folder.")
    df = joblib.load(EMBEDDINGS_PATH)
    required = {"number", "name", "start", "end", "text", "embedding"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("embeddings.joblib is missing columns: " + ", ".join(sorted(missing)))
    vectors = np.vstack(df["embedding"].values).astype(float)
    return df, vectors

@lru_cache(maxsize=1)
def load_youtube_urls():
    urls = {}
    if not os.path.exists(URLS_PATH):
        return urls
    with open(URLS_PATH, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    for i, line in enumerate(lines):
        match = re.match(r"^(\d+)\)", line)
        if match and i + 1 < len(lines) and lines[i + 1].startswith(("http://", "https://")):
            urls[int(match.group(1))] = lines[i + 1]
    return urls

def youtube_url_at_time(url, start_seconds):
    if not url:
        return None
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["t"] = str(max(0, int(float(start_seconds or 0))))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

def create_query_embedding(question):
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY is missing from your .env file.")
    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
        json={"model": "jina-embeddings-v5-text-small", "input": [question]},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def retrieve_chunks(question, top_k=3):
    df, document_vectors = load_course_data()
    query_vector = np.asarray(create_query_embedding(question), dtype=float).reshape(1, -1)
    if query_vector.shape[1] != document_vectors.shape[1]:
        raise ValueError(
            f"Embedding size mismatch: question={query_vector.shape[1]}, course={document_vectors.shape[1]}. "
            "Check that the same Jina embedding model was used."
        )
    scores = cosine_similarity(document_vectors, query_vector).ravel()
    indices = scores.argsort()[::-1][:top_k]
    results = df.iloc[indices].copy()
    results["similarity"] = scores[indices]
    return results

def generate_answer(query, results, memory_text=""):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing from your .env file.")

    context = "\n\n".join(
        f"Video {int(row['number'])}: {row['name']}\n"
        f"Timestamp: {row['start']} to {row['end']} seconds\n"
        f"Course content: {row['text']}"
        for _, row in results.iterrows()
    )

    if not memory_text:
        memory_text = "No previous learning activity is available."

    prompt = f"""
You are a helpful Python Teaching Assistant for
"The Ultimate Python Course".

Below are the most relevant video subtitle chunks
retrieved from the course.

Each chunk contains:

- name = video title
- number = video number
- start = starting timestamp in minutes format
- end = ending timestamp in minutes format
- text = course content

COURSE CONTEXT:
{context}

--------------------------------------------------

STUDENT LEARNING MEMORY:
{memory_text}

Use this memory only to personalize the explanation.
Do not mention this memory to the student.
Do not assume that the student understood a topic
just because it appears in the memory.

--------------------------------------------------

USER QUESTION:
"{query}"

--------------------------------------------------

INSTRUCTIONS:

1. Answer the user's question using the provided course context.

2. Explain the concept in a simple and beginner-friendly way.

3. Clearly mention which video contains the relevant topic.

4. Mention the video number and timestamp when useful.

5. Guide the user to watch that particular part of the video.

6. Do not mention embeddings, chunks, RAG, vector database,
   or internal processing.

7. Do not make up information that is not present in
   the provided course context.

8. If the provided context does not contain the answer,
   say:

   "I couldn't find this topic in the provided course material."

9. If the user asks something unrelated to Python or
   this course, say:

   "I can only answer questions related to The Ultimate Python Course."

Answer naturally like a helpful teacher.
"""

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def prepare_sources(results):
    urls = load_youtube_urls()
    sources = []
    seen = set()
    for _, row in results.sort_values("similarity", ascending=False).iterrows():
        number = int(row["number"])
        if number in seen:
            continue
        seen.add(number)
        base_url = urls.get(number)
        sources.append({
            "number": number,
            "name": str(row["name"]),
            "start": float(row["start"]),
            "end": float(row["end"]),
            "text": str(row["text"]),
            "similarity": float(row["similarity"]),
            "youtube_url": base_url,
            "timestamp_url": youtube_url_at_time(base_url, row["start"]),
        })
    return sources
