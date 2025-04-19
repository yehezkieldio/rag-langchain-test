import { Document } from "@langchain/core/documents";

// Format retrieved documents into a string for the LLM context
export function formatDocumentsAsString(docs: Document[]): string {
    return docs
        .map(
            (doc, index) =>
                `--- Document ${index + 1} ---\n${doc.pageContent}\nMetadata: ${JSON.stringify(doc.metadata)}`
        )
        .join("\n\n");
}
