# RAG App

A document-based Retrieval-Augmented Generation (RAG) application built with Python, FastAPI, and PostgreSQL.

The project focuses on building a practical RAG pipeline that enables users to upload documents, process and store their content, retrieve relevant information using semantic search, and generate context-aware answers using Large Language Models (LLMs).

## Current Implementation

The current implementation covers the core components of a RAG application:

- **REST API:** Building application endpoints using FastAPI.
- **Document Upload:** Uploading documents through the API.
- **File Processing:** Extracting and processing document content.
- **Data Pipeline:** Organizing document processing and indexing workflows.
- **Database Integration:** Working with MongoDB and migrating to PostgreSQL.
- **Database Models:** Managing database schemas using SQLAlchemy.
- **Database Migrations:** Handling schema changes with Alembic.
- **LLM Factory:** Creating a flexible interface for integrating different LLM providers.
- **Vector DB Factory:** Structuring vector database integrations.
- **Semantic Search:** Finding relevant document content using vector similarity.
- **Augmented Generation:** Generating context-aware answers using retrieved document information.
- **PgVector:** Storing and searching vector embeddings directly in PostgreSQL.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10 | Backend development |
| FastAPI | REST API |
| PostgreSQL | Relational database |
| PgVector | Vector storage and similarity search |
| SQLAlchemy | Database ORM |
| Alembic | Database migrations |
| MongoDB | Previous database implementation |
| Docker | Containerized services |
| Ollama | Local LLM integration |

## Getting Started

### Prerequisites

- Python 3.10
- Miniconda or another Python environment manager
- Docker and Docker Compose
- Required API credentials (depending on the selected LLM provider)

### 1. Install System Dependencies

```bash
sudo apt update
sudo apt install libpq-dev gcc python3-dev
```

### 2. Create a Python Environment

Using Conda:

```bash
conda create -n rag-app python=3.10
conda activate rag-app
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create your environment file:

```bash
cp .env.example .env
```

Update `.env` with your required configuration and credentials, such as your API keys and database connection settings.

### 5. Start Docker Services

Navigate to the Docker directory:

```bash
cd docker
cp .env.example .env
```

Configure the Docker environment variables, then start the services:

```bash
docker compose up -d
```

### 6. Run Database Migrations

Return to the project root directory:

```bash
cd ..
```

Apply database migrations:

```bash
alembic upgrade head
```

### 7. Run the Application

Start the FastAPI development server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

The application will be available at:

- **API:** http://localhost:5000
- **Swagger Documentation:** http://localhost:5000/docs
- **ReDoc:** http://localhost:5000/redoc

---

# Future Direction

## Knowledge Copilot — Agentic RAG Platform

The long-term goal is to evolve this project into a more advanced Agentic RAG platform called **Knowledge Copilot**.

The platform will move beyond traditional RAG by introducing intelligent retrieval strategies, agent-based workflows, and improved answer reliability.

### Planned Architecture & Features

**1. Advanced Document Retrieval**
- Build a document-based RAG platform using FastAPI, PostgreSQL, and PgVector.
- Support document ingestion, chunking, embedding, and retrieval across multiple documents.

**2. Hybrid Search & Reranking**
- Combine semantic vector search with keyword-based retrieval.
- Implement result fusion to combine retrieval results.
- Introduce reranking to improve retrieval relevance and accuracy.

**3. Agentic RAG with LangGraph**
- Build an Agentic RAG workflow using LangGraph.
- Implement query rewriting to improve search queries.
- Add conditional routing to determine the appropriate retrieval path.
- Introduce evidence assessment to evaluate retrieved information.
- Implement bounded retrieval retries when the available evidence is insufficient.
- Generate source-grounded answers based on retrieved context.

### Planned Tech Stack

- Python
- FastAPI
- PostgreSQL
- PgVector
- LangChain
- LangGraph
- LangSmith
- Docker

---

## Project Status

**Current Stage:** Core RAG Implementation

The project currently covers the foundational RAG pipeline, including document processing, database integration, LLM integration, semantic search, and PgVector.

The next development stages will focus on advanced retrieval, hybrid search, reranking, and Agentic RAG workflows.

