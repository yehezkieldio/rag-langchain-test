import { Document } from "@langchain/core/documents";
import type { ChatMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { RunnableBranch, RunnableLambda, RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { BufferMemory } from "langchain/memory";
import { llm, routerLlm } from "#/lib/llm";
import { formatDocumentsAsString } from "#/lib/utils";
import { hybridSearch } from "#/lib/vector-store";

// --- Router Chain ---
// Decides whether to use RAG or just respond conversationally

const routingPrompt = ChatPromptTemplate.fromMessages([
    [
        "system",
        `You are an expert classification system. Your goal is to determine if a user query requires accessing a knowledge base or if it's a general conversational request (like chit-chat, greetings, or asking about your capabilities without needing specific external data).

    Respond ONLY with one of the following words:
    'KB_QUERY' - if the query asks for specific information, facts, details likely found in documents.
    'CONVERSATIONAL' - if the query is a greeting, small talk, a general question about the AI itself, or doesn't require specific external facts.

    Examples:
    User: Tell me about the performance optimizations in v2.3.  Response: KB_QUERY
    User: What is the capital of France? Response: CONVERSATIONAL (General knowledge, doesn't need *our* KB)
    User: Hi there! Response: CONVERSATIONAL
    User: Summarize the main points of the last meeting. Response: KB_QUERY
    User: How are you today? Response: CONVERSATIONAL
    User: Explain the concept of hybrid search. Response: KB_QUERY (Assuming KB has info on it)
    User: Can you help me write an email? Response: CONVERSATIONAL`
    ],
    ["human", "User query: {query}"]
]);

// Use a potentially smaller/faster LLM for routing
const routerChain = routingPrompt.pipe(routerLlm).pipe(new StringOutputParser());

// --- RAG Chain ---
// Handles queries identified as needing the Knowledge Base

const ragPrompt = ChatPromptTemplate.fromMessages([
    [
        "system",
        `You are powered by a hybrid search engine. You can use both vector and keyword search to find relevant documents. Use the most relevant documents to answer the question.
        If asked for your model or what are you, say I am a DeepSeek V3 model, a hybrid search engine that combines vector and keyword search to find relevant documents.`
    ],
    [
        "system",
        `You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Context:
    {context}`
    ],

    new MessagesPlaceholder("chat_history"), // Include chat history
    ["human", "{query}"]
]);

const retrieveDocuments = async (input: { query: string; chat_history: ChatMessage[] }): Promise<Document[]> => {
    console.log(`Retrieving documents for: "${input.query}"`);
    return hybridSearch(input.query, 4); // Use hybrid search
};

const formatContext = (docs: Document[]): string => {
    return formatDocumentsAsString(docs);
};

const ragChain = RunnableSequence.from([
    RunnablePassthrough.assign({
        // Retrieve docs based on query
        context: new RunnableLambda({ func: retrieveDocuments }).pipe(formatContext)
    }),
    ragPrompt,
    llm, // Use the main LLM for generation
    new StringOutputParser()
]);

// --- Conversational Chain ---
// Handles queries identified as not needing the KB

const conversationalPrompt = ChatPromptTemplate.fromMessages([
    [
        "system",
        `If asked about your model or what you are, say I am a DeepSeek V3 model, a hybrid search engine that combines vector and keyword search to find relevant documents.`
    ],
    ["system", "You are a helpful and friendly general assistant. Respond clearly and concisely."],
    new MessagesPlaceholder("chat_history"), // Include chat history
    ["human", "{query}"]
]);

const conversationalChain = RunnableSequence.from([
    conversationalPrompt,
    llm, // Use the main LLM
    new StringOutputParser()
]);

// --- Main Chain with Router ---
// Uses RunnableBranch to direct the query based on the router's output

export const getMainChain = async () => {
    // Initialize memory (could use RedisChatMessageHistory for persistence across restarts)
    const memory = new BufferMemory({
        returnMessages: true,
        memoryKey: "chat_history",
        inputKey: "query", // Ensure memory knows which input key holds the human message
        outputKey: "output" // Optional: if you want to store AI output too
    });

    const chainWithMemory = RunnableSequence.from([
        // 1. Load history and pass query through
        RunnablePassthrough.assign({
            chat_history: new RunnableLambda({
                func: async (_input: { query: string }) => {
                    const history = await memory.loadMemoryVariables({});
                    return history.chat_history || []; // Return empty array if no history
                }
            })
        }),
        // 2. Route the query
        RunnableBranch.from([
            [
                async (x: { query: string; chat_history: ChatMessage[] }) => {
                    const res = await routerChain.invoke({ query: x.query });
                    console.log(`Router decision for "${x.query}": ${res}`);
                    return res.trim().toUpperCase() === "KB_QUERY";
                },
                ragChain
            ],
            conversationalChain // This becomes the default case
        ]),

        // 3. Save context AFTER the main chain runs
        new RunnableLambda({
            func: async (input: string, runnable: Record<string, unknown>) => {
                const parentRun = runnable as { query: string };
                if (parentRun.query) {
                    await memory.saveContext({ query: parentRun.query }, { output: input });
                }
                return input;
            }
        })
    ]);

    console.log("Running without Redis cache.");

    return chainWithMemory;
};
