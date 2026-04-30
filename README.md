# Cobb County Building & Fire Code Agentic RAG Assistant

Agentic RAG web app that answers Cobb County, Georgia building and fire code questions using local PDF documents first, with web search fallback when local evidence is insufficient or current-code verification is needed.

![Agentic RAG flow chart](assets/Rag%20Flow%20Chart.png)

> Core focus: Retrieval-Augmented Generation, agentic tool use, document indexing, source-grounded answers, and deployable ML application engineering.

---

## Project Highlights

- Task: Agentic Retrieval-Augmented Generation (RAG)
- Domain: Local government building, permitting, and fire code information
- Objective: Help users query complex Cobb County code documents through a simple chat interface
- Retrieval: Chroma vector database over local PDFs
- Agent behavior: Lightweight LLM router, local retrieval, evidence checks, and web fallback when current information is requested
- LLMs: OpenAI by default, optional Google Gemini
- App: Streamlit chat UI with source display and an explanatory "About the App" tab
- Deployment: Local Python, Docker Compose, and Streamlit Community Cloud compatible

---

## What This Project Does

Building and fire code information is often spread across ordinances, county PDFs, checklists, permit forms, and web pages. This app gives users a single chat interface that:

- Loads local Cobb County building and fire code PDFs
- Splits documents into searchable chunks
- Stores embeddings in a local Chroma vector database
- Retrieves relevant document excerpts for each user question
- Uses a lightweight LLM router to detect whether web search may be needed
- Checks whether local evidence is strong enough
- Uses web search when local retrieval is weak or when current code status needs verification
- Produces concise answers with source references

If the app cannot find reliable evidence, it responds clearly:

```text
I could not find a reliable answer in the available documents or web sources.
```

---

## Real-World Impact

This project demonstrates how RAG can reduce friction in document-heavy public-sector workflows.

Potential use cases include:

- Helping homeowners understand where to start with permit questions
- Supporting contractors who need to quickly locate relevant county guidance
- Assisting plan reviewers or administrative staff with document lookup
- Creating a searchable knowledge layer over local ordinances and PDF forms
- Demonstrating how AI systems can provide grounded answers instead of unsupported guesses

This is a portfolio demonstration, not an official Cobb County tool.

---

## Dataset Description

The local corpus is designed for Cobb County, Georgia building and fire code research.

Expected document types:

- Cobb County Code of Ordinances PDFs
- Cobb County Fire Marshal forms and checklists
- Building permit and tenant build-out guidance
- Fire inspection, hydrant, sprinkler, and emergency equipment documents
- Georgia state fire safety and construction code reference materials
- Current adopted code references from county or state sources

Current local development corpus:

| Dataset Item | Value |
|---|---:|
| Local PDF files used during development | 41 |
| Approximate raw PDF size | 60 MB |
| Loaded PDF pages | 4,093+ |
| Generated vector chunks | 13,844 |
| Public repo data policy | Raw PDFs excluded, vectorstore tracked with Git LFS |

The `data/README.md` file explains where users should place their own PDFs before rebuilding the vector index. The generated Chroma `vectorstore/` is tracked with Git LFS so Streamlit Community Cloud can load the demo index without private raw PDFs.

---

## Feature Engineering Overview

This project does not train a traditional tabular ML model. Instead, it engineers retrieval features for semantic search.

Key steps:

- PDF parsing: Each PDF is loaded page by page with source metadata
- Text chunking: Pages are split into overlapping chunks to preserve context
- Metadata tracking: File name, source path, and page number are retained
- Embeddings: Each chunk is converted into a dense vector representation
- Vector indexing: Chunks and metadata are stored in Chroma
- Query routing: A lightweight LLM classifier flags whether the query needs local retrieval, web search, or both
- Retrieval scoring: User questions are matched against indexed chunks
- Evidence thresholding: Weak retrieval triggers fallback web search

---

## Modeling Approach

The app uses an agentic RAG workflow rather than a single prompt-only LLM call.

```text
User question
    |
    v
Streamlit chat interface
    |
    v
Lightweight LLM query router
    |
    +--> Flags likely local retrieval, web search, or both
    |
    v
LangChain RAG controller
    |
    +--> Local retriever
    |       |
    |       v
    |   Chroma vector database over local PDF chunks
    |
    +--> Evidence quality check
            |
            +--> Strong local evidence: answer from documents
            |
            +--> Weak or current-code question: use web search
                    |
                    v
                Synthesize grounded answer with sources
```

Agent behavior:

- Uses a lightweight LLM router before retrieval
- Still retrieves local documents for Cobb County code questions
- Uses relevance scoring and an LLM adequacy check
- Keeps deterministic keyword/date routing as a backup
- Forces web verification for current, latest, adopted, or effective-date questions
- Keeps responses to 2-3 short paragraphs
- Shows whether the answer came from local documents, web search, or both
- Avoids legal, engineering, or permitting advice

---

## Results and Validation

This project was validated through ingestion, retrieval, and fallback behavior checks.

| Test Area | Result | Notes |
|---|---:|---|
| PDF ingestion | Passed | Loaded Cobb County and Georgia code PDFs |
| Vector index build | Passed | Indexed 13,844 chunks into Chroma |
| Local retrieval smoke test | Passed | Retrieved relevant fire inspection sources |
| LLM query router | Passed | Flags current, dated, and fee-schedule questions for web verification |
| Web search fallback | Passed | SerpAPI Google Search works from the app environment |
| Current-date sanity check | Passed | Runtime date context answers simple date questions |
| Current-code verification | Passed | Forces web search for currently adopted/effective code questions |
| App syntax check | Passed | `python -m compileall app src` |
| Docker support | Included | Dockerfile and docker-compose.yml |

Example retrieval test:

| Query | Expected Behavior | Observed Behavior |
|---|---|---|
| When is a fire inspection required? | Local retrieval | Returns Cobb County fire-related sources |
| What is today's date? | Runtime/web fallback | Answers using runtime date context |
| What are the currently adopted construction codes for Cobb County building permits? | Local + web verification | Uses local documents and Cobb County/state web sources |

---

## Key Insights

- RAG is a strong fit for code and permitting documents because answers must be source-grounded.
- Local retrieval alone is not enough for "current" or "effective date" questions because codes change.
- Keeping file and page metadata is essential for user trust.
- A conservative fallback response is safer than forcing an answer from weak evidence.
- Streamlit is effective for quickly turning a RAG pipeline into a recruiter-friendly demo app.
- Docker support improves reproducibility for portfolio reviewers and hiring managers.

---

## Tech Stack

- Language: Python 3.12
- App Framework: Streamlit
- RAG Framework: LangChain
- Vector Database: Chroma
- Embeddings: OpenAI by default, optional Gemini
- LLM: OpenAI by default, optional Google Gemini
- Web Search: SerpAPI Google Search
- PDF Loading: PyPDF / LangChain document loaders
- Deployment: Docker, Docker Compose, Streamlit Community Cloud
- Observability: Optional LangSmith tracing

---

## Project Structure

```text
.
├── app/
│   └── streamlit_app.py          # Streamlit chat UI and About tab
├── src/
│   ├── agent.py                  # Agentic RAG orchestration
│   ├── config.py                 # Environment and model configuration
│   ├── ingestion.py              # PDF loading, chunking, embedding, Chroma indexing
│   ├── retriever.py              # Chroma retrieval helpers and source formatting
│   ├── tools.py                  # Local retrieval and web search tools
│   └── check_web_search.py       # Web search diagnostic script
├── data/
│   └── README.md                 # Instructions for local PDF placement
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
├── .dockerignore
└── README.md
```

Excluded from GitHub:

- `.env`
- `secrets/`
- `notes/`
- `data/raw/`
- `.venv/`
- `.tiktoken_cache/`

Tracked with Git LFS:

- `vectorstore/**`

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/helsharif/cobb-county-code-rag-assistant.git
cd cobb-county-code-rag-assistant
```

### 2. Create a Python 3.12 virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Fill in your selected provider keys. Keep `.env` local only and never commit it:

```text
OPEN_API_KEY=<your-openai-key>
GEMINI_API_KEY=<optional-gemini-key>
SERPAPI_API_KEY=<your-serpapi-key>
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
```

Environment handling best practices used in this repo:

- `.env` is ignored by Git and excluded from Docker build context.
- `.env.example` is the public template for required and optional settings.
- The Docker image does not copy `.env` or bake secrets into image layers.
- `docker-compose.yml` maps only the variables the app expects instead of passing every local environment variable into the container.
- For shared, hosted, or production deployments, use the platform's secret manager instead of committing or baking secrets into Docker images.

### 5. Add local PDF documents

Place PDFs under:

```text
data/raw/
```

Example:

```text
data/raw/cobb_county_fire/
data/raw/cobb_municode/
data/raw/applicable_codes/
```

### 6. Build the vector index

```bash
python -m src.ingestion --rebuild
```

### 7. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

## Docker Run

Create a local `.env` from `.env.example` before running Docker Compose. Docker Compose reads this file for variable interpolation, but the image itself does not contain secrets.

Build and start the app:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8501
```

Rebuild the vector index inside Docker:

```bash
docker compose run --rm cobb-county-rag python -m src.ingestion --rebuild
```

---

## Example Questions

- What permits are required for residential construction in Cobb County?
- What are fire sprinkler requirements for commercial buildings?
- When is a fire inspection required?
- What are the currently adopted construction codes for Cobb County building permits, and when did they take effect?
- What is today's date?

---

## Screenshots and Figures

The Streamlit app includes an "About the App" tab with:

- High-level RAG architecture diagram
- Document ingestion flow chart
- Query-time retrieval and web fallback flow chart
- Tech stack table for non-technical reviewers

Suggested portfolio screenshots:

- Chat answer with local document sources
- Chat answer showing local + web fallback sources
- About tab architecture diagram

---

## Reproducibility and Best Practices

- Modular application code under `src/`
- Environment variables isolated in `.env`
- Public `.env.example` template
- No hardcoded secrets
- Raw PDFs excluded from Git
- Chroma vectorstore tracked with Git LFS for Streamlit Community Cloud deployment
- Rebuildable vector index
- Source metadata retained for citations
- Dockerized local deployment
- Streamlit Community Cloud compatible structure

---

## Disclaimer

This project is for portfolio demonstration and educational purposes only. It is not legal, engineering, building code, fire code, or permitting advice.

Users should verify all requirements directly with Cobb County, the Georgia Department of Community Affairs, the State Fire Marshal, and the relevant authority having jurisdiction.

---

## Author

Husayn El Sharif  
Senior Data Scientist / Machine Learning Engineer

---

## Portfolio Relevance

This project highlights:

- Applied RAG system design
- Agentic tool orchestration
- Vector search over local documents
- Web fallback for current information
- Source-grounded LLM responses
- Production-minded Streamlit and Docker deployment
- Public-sector AI workflow design
