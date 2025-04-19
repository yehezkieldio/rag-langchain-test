import fs from "node:fs/promises";
import path from "node:path";
import { Document } from "langchain/document";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { env } from "#/env";
import { getVectorStore, pool } from "#/lib/vector-store";

async function loadData() {
    console.log("Starting data loading process...");
    console.log(env.DATABASE_URL);

    try {
        // --- 1. Setup Vector Store (ensures table exists) ---
        const vectorStore = await getVectorStore();
        console.log("Vector store initialized.");

        // --- 2. Clear existing data (optional, for clean reload) ---
        console.log(`Clearing existing data from ${env.PG_COLLECTION_NAME}...`);
        await pool.query(`DELETE FROM ${env.PG_COLLECTION_NAME};`);
        console.log("Existing data cleared.");

        // --- 3. Load Documents ---
        const dataDir = path.join(__dirname, "..", "data");
        console.log(`Loading documents from directory: ${dataDir}`);

        // Get all .txt files in the data directory
        const files = await fs.readdir(dataDir);
        const txtFiles = files.filter((file) => file.endsWith(".txt"));

        if (txtFiles.length === 0) {
            console.log("No .txt files found in the data directory.");
            return;
        }

        let allDocs: Document[] = [];

        // Load each .txt file
        for (const file of txtFiles) {
            const filePath = path.join(dataDir, file);
            const loader = new TextLoader(filePath);
            const docs = await loader.load();
            allDocs = allDocs.concat(docs);
            console.log(`Loaded ${docs.length} document(s) from ${file}`);
        }

        console.log(`Loaded ${allDocs.length} total raw document(s).`);

        // --- 4. Split Documents ---
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 150
        });
        const splitDocs = await textSplitter.splitDocuments(allDocs);
        console.log(`Split into ${splitDocs.length} chunks.`);

        // --- 5. Add Metadata ---
        const docsWithMetadata = splitDocs.map((doc) => {
            doc.metadata = {
                ...doc.metadata,
                loaded_at: new Date().toISOString(),
                title: `Chunk from ${doc.metadata.source || "unknown"}`
            };
            return doc;
        });

        // Log a sample chunk with metadata
        if (docsWithMetadata.length > 0) {
            console.log("Sample chunk metadata:", docsWithMetadata);
        }

        // --- 6. Embed and Store Documents ---
        console.log("Embedding and adding documents to the vector store...");
        await vectorStore.addDocuments(docsWithMetadata);
        console.log("Documents added successfully!");
    } catch (error) {
        console.error("Error during data loading:", error);
    } finally {
        await pool.end();
        console.log("PostgreSQL pool closed.");
    }
}

loadData();
