import { ChatOpenAI } from "@langchain/openai";
import { env } from "#/env";

export const llm = new ChatOpenAI({
    model: env.OPENROUTER_LLM_MODEL,
    openAIApiKey: env.OPENROUTER_API_KEY,
    configuration: {
        baseURL: "https://openrouter.ai/api/v1"
    },
    temperature: 0.3,
    maxTokens: 500,
    streaming: false
});

export const routerLlm = new ChatOpenAI({
    modelName: "google/gemma-3-27b-it:free",
    openAIApiKey: env.OPENROUTER_API_KEY,
    configuration: {
        baseURL: "https://openrouter.ai/api/v1"
    },
    temperature: 0.0,
    maxTokens: 10
});
