import os
import re

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from fastembed import TextEmbedding

load_dotenv()

CHUNK_SIZE, OVERLAP = 1000, 200
TOP_K, MIN_SCORE = 5, 0.6
NOT_FOUND = "I couldn't find this information in the uploaded documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the provided document excerpts.
- Cite sources inline using only plain square brackets like [1], [2] matching the excerpt numbers.
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
- Use the conversation history to understand follow-up questions."""

REWRITE_PROMPT = """Rewrite the user's latest question as a standalone search query, resolving any references
to earlier messages (e.g. "it", "that company", "the previous year"). Return only the rewritten question."""


class RAG:
    def __init__(self):
        self.embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
        self.llm = Groq()
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.chunks, self.vectors, self.history = [], np.empty((0, 384)), []

    def add_pdf(self, file, name):
        new = []
        for page_no, page in enumerate(PdfReader(file).pages, 1):
            text = " ".join((page.extract_text() or "").split())
            for i in range(0, len(text), CHUNK_SIZE - OVERLAP):
                new.append({"text": text[i:i + CHUNK_SIZE], "source": name, "page": page_no})
        if new:
            self.chunks += new
            self.vectors = np.vstack([self.vectors, *self.embedder.embed([c["text"] for c in new])])
        return len(new)

    def retrieve(self, query):
        scores = self.vectors @ next(self.embedder.query_embed(query))
        return [self.chunks[i] for i in np.argsort(scores)[::-1][:TOP_K] if scores[i] >= MIN_SCORE]

    def chat(self, messages):
        res = self.llm.chat.completions.create(model=self.model, messages=messages, temperature=0)
        return re.sub(r"【(\d+)[^】]*】", r"[\1]", res.choices[0].message.content).strip()

    def ask(self, question):
        query = question
        if self.history:
            query = self.chat([{"role": "system", "content": REWRITE_PROMPT}, *self.history[-6:],
                               {"role": "user", "content": question}])

        sources = self.retrieve(query)
        if sources:
            context = "\n\n".join(f"[{i}] ({s['source']}, page {s['page']})\n{s['text']}" for i, s in enumerate(sources, 1))
            answer = self.chat([{"role": "system", "content": SYSTEM_PROMPT}, *self.history[-6:],
                                {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"}])
        else:
            answer = NOT_FOUND
        if NOT_FOUND in answer:
            sources = []

        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        return answer, sources

    def reset(self):
        self.history = []
