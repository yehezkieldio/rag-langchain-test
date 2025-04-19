import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { env } from "#/env";

export const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: env.HF_API_TOKEN,
    model: env.EMBEDDING_MODEL_NAME
});

export async function getEmbeddingDimension(): Promise<number> {
    if (env.EMBEDDING_MODEL_NAME.includes("MiniLM-L6-v2")) {
        return 384;
    }

    console.warn(
        "Warning: Could not automatically determine embedding dimension. Assuming 384. Set manually if incorrect."
    );
    return 384;
}
