# Hierarchical RAG Assistant Documentation

## Overview
This project implements a sophisticated Retrieval-Augmented Generation (RAG) system using LangChain.js, combining the power of large language models with efficient document retrieval. The system is designed to provide intelligent responses by leveraging both pre-loaded knowledge and real-time context.

## Architecture Deep Dive

### 1. Core Components

#### 1.1 Document Processing Pipeline (`loader.ts`)
The document processing pipeline is the foundation of our RAG system, responsible for ingesting and preparing documents for retrieval. Here's how it works:

```mermaid
graph TD
    A[Load Documents] --> B[Split Documents]
    B --> C[Add Metadata]
    C --> D[Generate Embeddings]
    D --> E[Store in Vector DB]
```

- **Document Loading**: Uses LangChain's `TextLoader` to read `.txt` files from the `data` directory
- **Text Splitting**: Implements `RecursiveCharacterTextSplitter` with:
  - `chunkSize: 1000` - Optimal balance between context and precision
  - `chunkOverlap: 150` - Ensures context continuity across chunks
- **Metadata Enrichment**: Adds temporal and source information to each chunk
- **Vector Storage**: Stores document embeddings in PostgreSQL for efficient similarity search

#### 1.2 Main Application Flow (`index.ts`)
The main application implements a robust chat interface with graceful shutdown handling:

- **Signal Handling**: Manages SIGINT, SIGTERM, and SIGUSR2 for clean shutdowns
- **Graceful Shutdown**: Implements a 5-second timeout for resource cleanup
- **Interactive Chat Loop**: Uses `inquirer` for a responsive CLI interface
- **Error Handling**: Comprehensive error management with proper cleanup

#### 1.3 Environment Configuration (`env.ts`)
Uses `@t3-oss/env-core` for type-safe environment variable management:

- **API Keys**: OpenRouter and HuggingFace integration
- **Database Configuration**: PostgreSQL connection details
- **Redis Configuration**: For caching and session management
- **Runtime Validation**: Zod schema validation for all environment variables

### 2. Library Components (`/lib`)

#### 2.1 Chain Implementation (`chain.ts`)
The chain implementation is the core of the RAG system, featuring a sophisticated routing mechanism:

```mermaid
graph TD
    A[User Query] --> B[Router Chain]
    B --> C{Query Type}
    C -->|KB Query| D[RAG Chain]
    C -->|Conversational| E[Conversational Chain]
    D --> F[Response]
    E --> F
```

##### Router Chain
- **Purpose**: Determines whether a query needs knowledge base access
- **Implementation**: Uses a specialized prompt with binary classification
- **Model**: Uses a smaller, faster model (Gemma-3-27b) for efficiency
- **Output**: Classifies queries as either 'KB_QUERY' or 'CONVERSATIONAL'

##### RAG Chain
- **Purpose**: Handles knowledge-intensive queries
- **Components**:
  - Document retrieval using hybrid search
  - Context formatting
  - Response generation with context
- **Memory**: Maintains conversation history for context

##### Conversational Chain
- **Purpose**: Handles general queries without KB access
- **Features**:
  - Maintains conversation history
  - Provides friendly, concise responses
  - No document retrieval overhead

#### 2.2 Vector Store (`vector-store.ts`)
Implements a sophisticated hybrid search system:

##### PostgreSQL Integration
- **Table Structure**:
  ```sql
  CREATE TABLE collection_name (
      id UUID PRIMARY KEY,
      content TEXT,
      metadata JSONB,
      embedding VECTOR(384)
  );
  ```
- **Indexes**:
  - HNSW index for vector similarity search
  - GIN index for full-text search
  - Metadata-specific indexes for structured queries

##### Hybrid Search Implementation
- **Vector Search**: Uses cosine similarity with HNSW indexing
- **Full-Text Search**: Implements PostgreSQL's tsvector/tsquery
- **Reciprocal Rank Fusion (RRF)**:
  - Combines vector and FTS results
  - Normalizes scores from both sources
  - Uses rank-based scoring for fair combination
  - Configurable decay parameter (K=60)

#### 2.3 Embeddings (`embeddings.ts`)
- **Model**: HuggingFace Inference API
- **Configuration**:
  - Dynamic dimension detection
  - Fallback to standard dimensions
  - API key management

#### 2.4 LLM Configuration (`llm.ts`)
- **Main LLM**: OpenRouter integration
  - Temperature: 0.3 (balanced creativity)
  - Max tokens: 500
  - Streaming: Disabled for stability
- **Router LLM**: Gemma-3-27b
  - Temperature: 0.0 (deterministic)
  - Max tokens: 10 (minimal output)

#### 2.5 Utilities (`utils.ts`)
- **Document Formatting**: Converts retrieved documents into LLM-friendly strings
- **Metadata Handling**: Preserves and formats document metadata

### 3. Key Technical Concepts

#### 3.1 Vector Store Implementation
The system uses PostgreSQL as a vector store, enabling:
- Efficient similarity search using vector embeddings
- Scalable document storage and retrieval
- Transaction support for data consistency

#### 3.2 Document Chunking Strategy
The text splitting approach is carefully designed to:
- Maintain semantic coherence within chunks
- Optimize for retrieval accuracy
- Balance context window limitations

#### 3.3 Error Handling and Recovery
Implements multiple layers of error handling:
- Process-level signal handling
- Database connection management
- Graceful degradation during failures

## Getting Started

### Prerequisites
- Node.js (v16 or higher)
- PostgreSQL database
- Redis instance
- Required API keys:
  - OpenRouter API key
  - HuggingFace API token

### Environment Setup
Create a `.env` file with the following variables:
```env
NODE_ENV=development
OPENROUTER_API_KEY=your_key
OPENROUTER_LLM_MODEL=your_model
HF_API_TOKEN=your_token
EMBEDDING_MODEL_NAME=your_model
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=your_db
DATABASE_URL=postgresql://user:password@host:port/db
PG_COLLECTION_NAME=your_collection
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_URL=redis://localhost:6379
```

### Running the Application
1. Install dependencies:
   ```bash
   npm install
   ```

2. Load documents:
   ```bash
   npm run load
   ```

3. Start the application:
   ```bash
   npm start
   ```

## Best Practices and Considerations

### 1. Document Processing
- Keep document chunks between 500-1500 tokens for optimal retrieval
- Use meaningful metadata to enhance search capabilities
- Regularly update the document store to maintain relevance

### 2. Performance Optimization
- Implement caching for frequently accessed documents
- Use connection pooling for database operations
- Monitor vector store performance and adjust chunk sizes accordingly

### 3. Security Considerations
- Secure storage of API keys and credentials
- Implement rate limiting for API calls
- Regular security audits of dependencies

## Troubleshooting

### Common Issues
1. **Vector Store Connection Issues**
   - Verify PostgreSQL connection settings
   - Check network connectivity
   - Ensure proper table initialization

2. **Document Loading Failures**
   - Verify file permissions
   - Check file encoding (UTF-8 recommended)
   - Monitor memory usage during large document processing

3. **API Integration Problems**
   - Validate API keys
   - Check rate limits
   - Verify model availability

## Contributing
When contributing to this project:
1. Follow the existing code style
2. Add comprehensive error handling
3. Update documentation for new features
4. Include tests for new functionality

## License
[Your License Here]
