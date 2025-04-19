import path from "node:path";
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
        // Example: Loading from a simple text file in the 'data' directory
        const dataPath = path.join(__dirname, "..", "data", "sample.txt"); // Adjust path as needed
        console.log(`Loading documents from: ${dataPath}`);
        const loader = new TextLoader(dataPath);
        const rawDocs = await loader.load();

        if (!rawDocs || rawDocs.length === 0) {
            console.log("No documents found to load.");
            return;
        }
        console.log(`Loaded ${rawDocs.length} raw document(s).`);
        // Add more loaders here for different file types (JSONLoader, CSVLoader, etc.)

        // --- 4. Split Documents ---
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000, // Adjust size as needed
            chunkOverlap: 150 // Adjust overlap as needed
        });
        const splitDocs = await textSplitter.splitDocuments(rawDocs);
        console.log(`Split into ${splitDocs.length} chunks.`);

        // --- 5. Add Metadata (Example) ---
        // You can add source, timestamps, keywords etc. here
        const docsWithMetadata = splitDocs.map((doc) => {
            // Example: Add source filename to metadata
            doc.metadata = {
                ...doc.metadata, // Keep existing metadata from loader
                source: path.basename(dataPath),
                loaded_at: new Date().toISOString(),
                // Add a dummy 'title' if your source doesn't have one, for FTS example
                title: `Chunk from ${path.basename(dataPath)}`
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
        // Close the pool connection if the script is standalone
        await pool.end();
        console.log("PostgreSQL pool closed.");
    }
}

loadData();
