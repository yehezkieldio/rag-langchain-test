import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { Document } from "@langchain/core/documents";
import { Pool, type PoolConfig } from "pg";
import { env } from "#/env";
import { embeddings, getEmbeddingDimension } from "#/lib/embeddings";

const pgConfig: PoolConfig = {
    connectionString: env.DATABASE_URL
};
export const pool = new Pool(pgConfig);

let vectorStoreInstance: PGVectorStore | null = null;

export async function getVectorStore(forceNew = false): Promise<PGVectorStore> {
    if (vectorStoreInstance && !forceNew) {
        return vectorStoreInstance;
    }

    await ensureTableExists();

    vectorStoreInstance = await PGVectorStore.initialize(embeddings, {
        pool,
        tableName: env.PG_COLLECTION_NAME,
        columns: {
            idColumnName: "id",
            vectorColumnName: "embedding",
            contentColumnName: "content",
            metadataColumnName: "metadata"
        },
        distanceStrategy: "cosine"
    });

    return vectorStoreInstance;
}

export async function ensureTableExists(): Promise<void> {
    const client = await pool.connect();
    try {
        await client.query("CREATE EXTENSION IF NOT EXISTS vector;");

        const dimension = await getEmbeddingDimension(); // Get dimension dynamically

        await client.query(`
            CREATE TABLE IF NOT EXISTS ${env.PG_COLLECTION_NAME} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT,
                metadata JSONB,
                embedding VECTOR(${dimension}) -- Specify dimension
            );
        `);

        // Index for Vector Search (HNSW is often good, adjust parameters as needed)
        await client.query(`
            CREATE INDEX IF NOT EXISTS ${env.PG_COLLECTION_NAME}_embedding_idx
            ON ${env.PG_COLLECTION_NAME}
            USING HNSW (embedding vector_cosine_ops); -- Use cosine distance
        `);

        // Index for Full-Text Search on the 'content' column
        await client.query(`
            CREATE INDEX IF NOT EXISTS ${env.PG_COLLECTION_NAME}_content_fts_idx
            ON ${env.PG_COLLECTION_NAME}
            USING GIN (to_tsvector('english', content));
        `);

        // Index for Full-Text Search on specific metadata fields (example: title)
        // Useful if you store structured metadata you want to search
        await client.query(`
            CREATE INDEX IF NOT EXISTS ${env.PG_COLLECTION_NAME}_metadata_fts_idx
            ON ${env.PG_COLLECTION_NAME}
            USING GIN (to_tsvector('english', metadata->>'title')); -- Example: Indexing a 'title' field
        `);

        console.log(`Table ${env.PG_COLLECTION_NAME} and indexes ensured.`);
    } catch (err) {
        console.error("Error ensuring table exists:", err);
        throw err;
    } finally {
        client.release();
    }
}

export async function fullTextSearch(query: string, k = 4): Promise<Document[]> {
    const client = await pool.connect();
    try {
        // Use plainto_tsquery for safer query parsing
        // Use ts_rank_cd for relevance scoring (BM25 approximation)
        const queryText = `
            SELECT
                id,
                content,
                metadata,
                ts_rank_cd(to_tsvector('english', content || ' ' || coalesce(metadata->>'title', '')), plainto_tsquery('english', $1)) AS score
            FROM ${env.PG_COLLECTION_NAME}
            WHERE to_tsvector('english', content || ' ' || coalesce(metadata->>'title', '')) @@ plainto_tsquery('english', $1)
            ORDER BY score DESC
            LIMIT $2;
        `;
        // Combine content and metadata title for searching

        const res = await client.query(queryText, [query, k]);

        return res.rows.map(
            (row) =>
                new Document({
                    pageContent: row.content,
                    metadata: { ...row.metadata, id: row.id, score: row.score } // Include score and id
                })
        );
    } catch (err) {
        console.error("Error during FTS search:", err);
        return [];
    } finally {
        client.release();
    }
}

// Simple Hybrid Search: Fetch Vector + FTS results and combine (Reciprocal Rank Fusion)
export async function hybridSearch(query: string, k = 4): Promise<Document[]> {
    const vectorStore = await getVectorStore();

    // 1. Perform Vector Search
    const vectorResults = await vectorStore.similaritySearchWithScore(query, k);
    // PGVectorStore returns distance (lower is better), we need score (higher is better)
    const vectorDocs = vectorResults.map(([doc, score]) => ({
        ...doc,
        // Normalize score (0 to 1), higher is better. Cosine distance is 0 to 2.
        // Simple inversion: score = 1 - (distance / 2)
        metadata: { ...doc.metadata, vector_score: 1 - score / 2, source: "vector" }
    }));

    // 2. Perform Full-Text Search
    const ftsDocsRaw = await fullTextSearch(query, k);
    // Normalize FTS score (ts_rank_cd often needs scaling) - this is heuristic
    const maxFtsScore = ftsDocsRaw[0]?.metadata?.score || 1.0; // Avoid division by zero
    const ftsDocs = ftsDocsRaw.map((doc) => ({
        ...doc,
        metadata: { ...doc.metadata, fts_score: doc.metadata.score / (maxFtsScore + 1e-6), source: "fts" } // Normalize
    }));

    // 3. Combine and Re-rank (Reciprocal Rank Fusion - RRF)
    const allDocs: { [id: string]: { doc: Document; score: number } } = {};
    const rrfK = 60; // RRF K parameter (controls score decay)

    const processResults = (results: Document[], type: "vector" | "fts") => {
        results.forEach((doc, rank) => {
            const id = doc.metadata.id || JSON.stringify(doc.pageContent).substring(0, 50); // Need a unique ID
            const score = 1 / (rrfK + rank + 1); // RRF score based on rank

            if (!allDocs[id]) {
                allDocs[id] = { doc: doc, score: 0 };
            }
            // Add scores from different sources
            allDocs[id].score += score;
            // Keep track of original scores/sources if needed
            allDocs[id].doc.metadata[`${type}_rank`] = rank + 1;
        });
    };

    processResults(vectorDocs, "vector");
    processResults(ftsDocs, "fts");

    // Sort by combined RRF score
    const combinedResults = Object.values(allDocs)
        .sort((a, b) => b.score - a.score)
        .slice(0, k) // Return top K results
        .map((item) => item.doc); // Return only the documents

    console.log(
        `Hybrid Search: Vector found ${vectorDocs.length}, FTS found ${ftsDocs.length}, Combined ${combinedResults.length}`
    );
    // console.log("Combined results (Top 1):", combinedResults[0]?.metadata);

    return combinedResults;
}
