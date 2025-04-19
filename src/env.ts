import { createEnv } from "@t3-oss/env-core";
import { z } from "zod";

export const env = createEnv({
    server: {
        NODE_ENV: z.enum(["development", "production"]),

        OPENROUTER_API_KEY: z.string(),
        OPENROUTER_LLM_MODEL: z.string(),

        HF_API_TOKEN: z.string(),
        EMBEDDING_MODEL_NAME: z.string(),

        POSTGRES_HOST: z.string(),
        POSTGRES_PORT: z.coerce.number(),
        POSTGRES_USER: z.string(),
        POSTGRES_PASSWORD: z.string(),
        POSTGRES_DB: z.string(),

        DATABASE_URL: z.string(),
        PG_COLLECTION_NAME: z.string(),

        REDIS_HOST: z.string(),
        REDIS_PORT: z.coerce.number(),
        REDIS_URL: z.string()
    },
    runtimeEnv: process.env
});
