import { Runnable } from "@langchain/core/runnables";
import inquirer from "inquirer"; // Use v8
import { getMainChain } from "#/lib/chain";
import { pool } from "#/lib/vector-store";

let isExiting = false;
let shutdownTimeout: NodeJS.Timeout;

async function gracefulShutdown(signal?: string) {
    if (isExiting) return;
    isExiting = true;

    console.log(`\nReceived ${signal || "shutdown"} signal. Shutting down...`);

    shutdownTimeout = setTimeout(() => {
        console.error("Forced shutdown after timeout");
        process.exit(1);
    }, 5000);

    try {
        await Promise.race([
            pool.end(),
            new Promise((_, reject) => setTimeout(() => reject(new Error("Database shutdown timeout")), 3000))
        ]);

        console.log("Connections closed. Exiting.");
        clearTimeout(shutdownTimeout);
        process.exit(0);
    } catch (error) {
        console.error("Error during shutdown:", error);
        clearTimeout(shutdownTimeout);
        process.exit(1);
    }
}

process.on("SIGINT", () => gracefulShutdown("SIGINT"));
process.on("SIGTERM", () => gracefulShutdown("SIGTERM"));
process.on("SIGUSR2", () => gracefulShutdown("SIGUSR2"));
process.on("unhandledRejection", (reason) => {
    console.error("Unhandled Rejection:", reason);
    gracefulShutdown("unhandled rejection");
});

async function main() {
    console.log("Initializing...");

    let mainChain: Runnable<unknown, string>;
    try {
        mainChain = await getMainChain();
        console.log("Main chain initialized.");
    } catch (error) {
        console.error("Failed to initialize the main chain:", error);
        await gracefulShutdown();
        return;
    }

    console.log("\nWelcome to the Hierarchical RAG Assistant!");
    console.log("Ask a question about the loaded documents or have a general chat.");
    console.log("Type 'exit' or press Ctrl+C to quit.\n");

    const chatLoop = async () => {
        while (!isExiting) {
            try {
                interface PromptResult {
                    query: string;
                }

                // Add timeout to inquirer prompt to allow interruption
                const answer = await Promise.race<PromptResult>([
                    inquirer.prompt([
                        {
                            type: "input",
                            name: "query",
                            message: "You:",
                            validate: (_input: string) => {
                                if (isExiting) {
                                    return false; // Break validation if shutting down
                                }
                                return true;
                            }
                        }
                    ]),
                    new Promise((_, reject) => {
                        const checkExit = setInterval(() => {
                            if (isExiting) {
                                clearInterval(checkExit);
                                reject(new Error("Shutdown requested"));
                            }
                        }, 100);
                    })
                ]);

                if (isExiting) break;

                const query = answer?.query?.trim();

                if (query?.toLowerCase() === "exit") {
                    await gracefulShutdown();
                    break;
                }

                if (!query) {
                    continue;
                }

                console.log("Assistant working...");

                const result = await mainChain.invoke({ query });

                console.log(`\nAssistant:\n${result}\n`);
            } catch (error) {
                if (isExiting) break; // Break the loop if we're shutting down
                console.error("\nAn error occurred:", error);
            }
        }
    };

    await chatLoop();
}

main().catch(async (err) => {
    console.error("Unhandled error in main function:", err);
    await gracefulShutdown(); // Ensure cleanup on unhandled error
});
