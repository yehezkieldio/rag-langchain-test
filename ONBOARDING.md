<!-- filepath: /home/elizielx/Documents/Temporary/rag-langchain-test/ONBOARDING.md -->
# Project Onboarding: RAG LangChain Test

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain.js. It provides an interactive command-line interface (CLI) where a user can chat with an AI assistant. The assistant can answer questions based on information loaded into a knowledge base or engage in general conversation.

A key feature is its use of **hybrid search**, combining semantic vector search with traditional keyword-based full-text search (FTS) for more robust and relevant document retrieval from the knowledge base.

## Core Components & Workflow

The system is built around several key components and follows a specific workflow:

1.  **Environment & Setup:**
    *   **Runtime:** Uses `bun` for package management and execution.
    *   **Database:** Requires a PostgreSQL database with the `pgvector` extension. A `docker-compose.yml` file is provided to easily set up a suitable container (`pgvector/pgvector:pg17`).
    *   **Dependencies (`package.json`):**
        *   `langchain` libraries (`@langchain/core`, `@langchain/community`, `@langchain/openai`): For building the core RAG logic, prompts, chains, and LLM interactions.
        *   `pg`, `@langchain/community/vectorstores/pgvector`: For PostgreSQL database interaction and managing the vector store.
        *   `@huggingface/inference`: To generate text embeddings via the Hugging Face API.
        *   `@t3-oss/env-core`, `zod`: For type-safe environment variable management (see `src/env.ts`).
        *   `inquirer`: For the interactive CLI prompt.
    *   **Configuration:** Requires API keys for OpenRouter (`OPENROUTER_API_KEY`) and Hugging Face (`HF_API_TOKEN`), plus database connection details, defined in environment variables (likely via a `.env` file, managed by `src/env.ts`).

2.  **Data Loading (`src/loader.ts`):**
    *   This script prepares and loads data into the knowledge base (PostgreSQL).
    *   **Initialization:** Ensures the database table (defined by `env.PG_COLLECTION_NAME`) and the `vector` extension exist (`src/lib/vector-store.ts::ensureTableExists`).
    *   **Loading:** Reads source documents (e.g., `data/sample.txt` using `TextLoader`).
    *   **Chunking:** Splits documents into smaller, manageable chunks (`RecursiveCharacterTextSplitter`).
    *   **Metadata:** Adds relevant metadata (source file, load time, etc.) to each chunk.
    *   **Embedding & Storage:** Generates vector embeddings for each chunk using a Hugging Face model (`src/lib/embeddings.ts`) and stores the chunk text, metadata, and embedding vector in the PostgreSQL table via `PGVectorStore`.

3.  **Embeddings (`src/lib/embeddings.ts`):**
    *   Configures the `HuggingFaceInferenceEmbeddings` client using the model specified in `env.EMBEDDING_MODEL_NAME` and the `HF_API_TOKEN`.
    *   Determines the embedding dimension needed for the database schema.

4.  **Vector Store & Hybrid Search (`src/lib/vector-store.ts`):**
    *   Manages the `PGVectorStore` instance and the underlying PostgreSQL connection pool (`pg`).
    *   `ensureTableExists`: Creates the table and necessary indexes:
        *   HNSW index for vector similarity search (cosine distance).
        *   GIN indexes for full-text search on content and specific metadata fields (e.g., 'title').
    *   `fullTextSearch`: Performs keyword search using PostgreSQL FTS functions (`to_tsvector`, `plainto_tsquery`, `ts_rank_cd`).
    *   `hybridSearch`:
        *   Performs both a vector similarity search and a full-text search.
        *   Normalizes scores from both search types.
        *   Combines and re-ranks results using Reciprocal Rank Fusion (RRF) to leverage the strengths of both methods.

5.  **Language Models (LLMs) (`src/lib/llm.ts`):**
    *   Configures `ChatOpenAI` clients to interact with LLMs via the OpenRouter API (`env.OPENROUTER_API_KEY`).
    *   `llm`: The primary model (`env.OPENROUTER_LLM_MODEL`) for generating final answers.
    *   `routerLlm`: A potentially smaller/faster model used specifically for classifying user intent.

6.  **Core Logic Chain (`src/lib/chain.ts`):**
    *   `getMainChain`: Constructs the main LangChain Runnable sequence that handles user interaction.
    *   **Memory:** Uses `BufferMemory` to retain conversation history (`chat_history`).
    *   **Router:**
        *   A prompt (`routingPrompt`) asks the `routerLlm` to classify the user's query as `KB_QUERY` (requires knowledge base) or `CONVERSATIONAL` (general chat).
        *   `RunnableBranch` directs the query based on the classification.
    *   **RAG Path (`KB_QUERY`):**
        *   Retrieves relevant documents using `hybridSearch`.
        *   Formats documents into a context string.
        *   Uses `ragPrompt` to instruct the main `llm` to answer based *only* on the provided context and chat history.
    *   **Conversational Path (`CONVERSATIONAL`):**
        *   Uses `conversationalPrompt` to instruct the main `llm` to respond as a general assistant, using chat history.
    *   **Output & Memory:** Returns the generated string response and updates the conversation memory.

7.  **Application Entry Point (`src/index.ts`):**
    *   Initializes the `mainChain`.
    *   Sets up graceful shutdown handlers to close the database connection pool cleanly.
    *   Runs an interactive loop using `inquirer` to get user input.
    *   Invokes the `mainChain` with the query.
    *   Prints the assistant's response.
    *   Handles the `exit` command and errors.

## How to Run

1.  **Prerequisites:**
    *   Install `bun`.
    *   Install Docker and Docker Compose.
    *   Create a `.env` file with necessary environment variables (see `src/env.ts` for required variables like `DATABASE_URL`, `OPENROUTER_API_KEY`, `HF_API_TOKEN`, `PG_COLLECTION_NAME`, `OPENROUTER_LLM_MODEL`, `EMBEDDING_MODEL_NAME`).
2.  **Start Database:** `docker compose up -d`
3.  **Install Dependencies:** `bun install`
4.  **Load Data:** `bun run src/loader.ts` (or the script defined in `package.json` for loading)
5.  **Run Chat:** `bun run src/index.ts` (or the main run script from `package.json`)

## Key Concepts

*   **RAG (Retrieval-Augmented Generation):** Enhancing LLM responses by first retrieving relevant information from an external knowledge base and providing it as context to the LLM during generation.
*   **Vector Embeddings:** Numerical representations of text that capture semantic meaning. Similar concepts have similar vectors.
*   **Vector Store:** A database optimized for storing and searching vector embeddings (here, PostgreSQL with `pgvector`).
*   **Full-Text Search (FTS):** Traditional keyword-based search.
*   **Hybrid Search:** Combining vector search (for semantic relevance) and FTS (for keyword matching) to improve retrieval accuracy.
*   **Reciprocal Rank Fusion (RRF):** A technique to combine ranked lists from different search methods (like vector and FTS) into a single, improved ranking.
*   **LangChain Runnables:** A composable interface for building chains of operations (retrieval, prompting, LLM calls, parsing).
*   **Routing:** Using an LLM or other logic to decide which processing path a query should take (e.g., RAG vs. conversational).

## RAG & Retrieval Improvements:

1. Advanced Re-ranking: While RRF is good, consider adding a cross-encoder re-ranking step after the initial hybrid retrieval. A cross-encoder takes the query and each retrieved document pair and provides a more accurate relevance score than vector/FTS alone. Libraries like sentence-transformers (via Python or a dedicated service) can be used for this.

2. Query Transformation: Before retrieval, transform the user's query to potentially improve retrieval results. Techniques include:
    - Hypothetical Document Embeddings (HyDE): Generate a hypothetical answer to the query first, embed that, and use the resulting vector for similarity search. This can sometimes align better with the embeddings of actual answer documents.
    - Multi-Query Retriever: Use an LLM to generate multiple variations of the user's query from different perspectives, perform retrieval for each, and combine the results.

3. Contextual Compression: Pass retrieved documents through a compression step before sending them to the LLM. This filters out irrelevant information within the retrieved documents, allowing you to potentially retrieve more documents initially (higher K) without exceeding context limits. LangChain offers `ContextualCompressionRetriever` wrappers.

4. Parent Document Retriever: Index smaller chunks for better retrieval accuracy, but retrieve larger parent documents (or surrounding chunks) associated with the matched small chunks to provide more context to the LLM. This helps avoid losing context at chunk boundaries.

5. Metadata Filtering: Allow filtering during retrieval based on metadata (e.g., source document, date range). This can be added as an option to the `hybridSearch` function in `vector-store.ts` or integrated directly if using LangChain retrievers that support it.

6. Chunking Strategy: Experiment with different chunking strategies in `loader.ts`. `RecursiveCharacterTextSplitter` is a good default, but semantic chunking (grouping text based on meaning using embeddings) might yield better results for some documents.

## Generation & Chain Improvements:

1. Refine Prompts:
    - In `chain.ts`, explicitly instruct the RAG prompt (ragPrompt) to cite the source documents (using the metadata you added) for its answers to improve trustworthiness.
    - Consider adding more sophisticated handling for cases where the context doesn't contain the answer, beyond just "I don't know".

2. Streaming: Modify the LLM calls and the chain in chain.ts and index.ts to support streaming responses. This significantly improves the perceived responsiveness in the CLI. Both ChatOpenAI and Runnables support streaming.

3. Persistent Chat History: Replace BufferMemory with a persistent store like RedisChatMessageHistory (you already have Redis in your docker-compose.yml and env.ts) to maintain conversation history across application restarts. This requires adding the redis package and updating the memory initialization in getMainChain.

## Evaluation & Observability:

RAG Evaluation: Implement evaluation metrics specific to RAG systems. Frameworks like RAGAs or LangChain's evaluation tools can help measure:
    - Context Precision/Recall: Is the retrieved context relevant and sufficient?
    - Faithfulness: Does the generated answer stick to the provided context?
    - Answer Relevance: Does the answer address the user's query