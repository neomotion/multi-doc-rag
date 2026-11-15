# Multi-Document RAG with Chat History and Conversational Memory

Upload several PDF reports and ask questions in plain language. Answers come only from the documents, cite the source file and page, say clearly when the information isn't there, and remember the conversation so follow-up questions work.

## Features

- **Multi-PDF upload**: index any number of PDFs from the Streamlit sidebar
- **Semantic retrieval**: `bge-small-en-v1.5` embeddings (via `fastembed`, runs locally on CPU) with cosine similarity
- **Grounded answers**: the LLM (Groq, `openai/gpt-oss-120b`) answers only from the retrieved excerpts
- **Source references**: inline citations like `[1]`, plus an expandable list of file name, page and excerpt
- **Unavailable information**: if the retrieved excerpts do not contain the answer, the LLM replies *"I couldn't find this information in the uploaded documents."*
- **Conversational memory**: follow-ups such as *"what about the previous year?"* are rewritten into a standalone search query using the chat history. The last 3 exchanges are also passed to the LLM.
- **Chat history**: shown in the UI, with a button to clear it

## How it works

```
PDFs ─► extract text per page ─► 1000-char chunks (200 overlap) ─► embeddings (in-memory)

Question + chat history ─► LLM rewrites to standalone query ─► top-5 most similar chunks
        ─► LLM answers from excerpts + history ─► answer with [n] citations + sources
                                                 (or "I couldn't find this information..." if not in excerpts)
```

## Sample documents

[`sample_docs/`](sample_docs) contains three public reports for testing:

| File | Type |
|---|---|
| `berkshire_2023_shareholder_letter.pdf` | Financial / annual report |
| `nist_cybersecurity_framework_2.0.pdf` | Technical framework |
| `nist_ai_risk_management_framework.pdf` | Technical / policy report |

Example conversation:

1. *What were Berkshire's operating earnings in 2023?* → answer cited to the Berkshire letter
2. *How does that compare with the previous year?* → memory resolves "that" to Berkshire's operating earnings
3. *What are the six functions of the NIST Cybersecurity Framework?* → answer cited to the CSF 2.0 document
4. *Which of them deals with recovering from incidents?* → follow-up answered from context
5. *What is the recipe for chocolate cake?* → "I couldn't find this information in the uploaded documents."

## How to run

1. Get a free API key at [console.groq.com](https://console.groq.com).
2. Install and run:

```bash
pip install -r requirements.txt
cp .env.example .env    # then put your GROQ_API_KEY in .env
streamlit run app.py
```

3. Upload PDFs in the sidebar (for example the ones in `sample_docs/`) and start asking questions.

You can change the model with `GROQ_MODEL` in `.env`.

## Project structure

```
rag.py          RAG class: PDF loading, chunking, embeddings, retrieval, query rewriting, answering, memory
app.py          Streamlit chat UI
sample_docs/    sample PDF reports
requirements.txt
.env.example
```
