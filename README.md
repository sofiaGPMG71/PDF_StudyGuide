# StudyPDF — LangChain + Streamlit PDF Study Assistant

Upload multiple lecture notes, textbooks or research papers, then ask questions,
request explanations, summarize a topic or generate quiz questions.
OpenAI is the default provider. No agent framework is needed for this RAG task.

## Run on Windows / VS Code (Python 3.11 or 3.12)

Extract this ZIP and open the StudyPDF folder in VS Code. Run in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

These commands do not require activating the environment. Keep requirements.txt
in the project folder, outside .venv. If Python 3.12 is installed, replace 3.11
with 3.12. Open the localhost URL printed by Streamlit.

1. Select OpenAI (default) and enter your own API key in the password field in
   the sidebar. The app uses the key for the current session and does not save
   it to a file.
2. Default models: `gpt-4o-mini` for answers, `text-embedding-3-small` for embeddings.
   Model names are editable if your account uses different available models.
3. Upload one or more PDFs and click **Process documents**.
4. Choose a study mode and type your request.
5. Check the numbered citations and expanded source chunks below each answer.

OpenAI API usage is billed separately from a ChatGPT subscription. Document text
is sent to OpenAI for embeddings; questions and retrieved excerpts are sent for
answer generation. The app does not read API keys from `.env` or save PDFs or
API keys to disk by itself.

If processing reports an OpenAI connection error, retry and check your internet
connection, VPN/firewall settings and OpenAI service status. The app retries
transient embedding failures; a connection error by itself does not confirm
whether the API key is valid. Never share the key in chat or screenshots. For
local processing, select Ollama and ensure its server and models are available.

## How it works

PDF bytes → pypdf page extraction → LangChain page-aware text splitting → OpenAI
embeddings → LangChain InMemoryVectorStore → cosine similarity retrieval →
LangChain chat messages → OpenAI grounded answer with [S1] source citations.

Chunks never cross PDF page boundaries. Metadata preserves filename, document
ID, physical PDF page (starting at 1), character offset and chunk ID. Printed
page labels may differ. Each assistant message stores its own exact source
passages, so later questions do not overwrite earlier evidence. Follow-up
questions are rewritten to standalone retrieval queries using recent chat.

Changing uploads, embedding settings, key or chunk settings blocks chat until
reprocessing. Successful reprocessing resets conversation. A failed process
keeps the prior index but does not enable it for changed settings. Indexes are
per Streamlit session and held in memory, not shared through a global cache.
Reset workspace releases the index; remove files in the uploader separately.

## Scope and limits

- Text-based PDFs supported; scanned/image-only pages need external OCR.
- Total upload limit: 50 MB; index limit: 5,000 chunks.
- Topic summaries and quizzes cover retrieved passages, not guaranteed full-book
  coverage. Ask focused questions for better results.
- Prompts require evidence and admit missing information. Citation ID checks flag
  missing/unknown IDs but cannot prove factual correctness. Always review sources.
- Similarity scores measure retrieval ranking, not answer confidence. Retrieval
  can miss relevant passages; no calibrated relevance threshold is imposed.
- Table layouts, equations and multi-column PDFs can extract imperfectly.
- In-memory search suits course projects; large collections need a persistent
  vector database and more scalable retrieval.
- This is a local course-project app, not a production multi-user deployment.

## Optional local alternative

Install Ollama, then run `ollama pull qwen3:4b` and
`ollama pull nomic-embed-text`. Select Ollama in the sidebar. This route does not
need an OpenAI key. The default local context window is 16,384 tokens; reduce
retrieved chunks if your model cannot handle long excerpts.

## Demo and checks

Upload `sample_lecture.pdf` and ask:
- What is overfitting? Explain simply.
- Summarize the differences between training, validation and test sets.
- Create three quiz questions with an answer key.
- What is the lecturer's email? (not in the sample; should admit missing evidence).

Run `python -m unittest discover -s tests -v` after installing dependencies.
Tests exercise real PDF extraction, page-preserving chunking, LangChain vector
retrieval with deterministic test embeddings, and source/citation handling with
a mock model. They do not call OpenAI or validate a live model's answer quality.

## Assignment requirements

| Requirement | Implementation |
|---|---|
| One or multiple PDF uploads | Streamlit multi-file uploader |
| Process documents | pypdf extracts text per page |
| Searchable representations | OpenAI embeddings in LangChain vector store |
| Retrieve relevant content | Top-k cosine similarity search |
| Grounded answers | Evidence-only system prompt with source IDs |
| Study assistance | Question, explanation, topic summary and quiz modes |
| Minimum challenge: show sources | Exact retrieved chunks, filenames and PDF pages alongside answers |

References checked during development:
- https://reference.langchain.com/python/langchain-core/vectorstores/in_memory/InMemoryVectorStore
- https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI
- https://docs.langchain.com/oss/python/integrations/providers/ollama/

Dependency versions are bounded to compatible major releases; install resolves
current minor releases. No live OpenAI call has been made during delivery.
