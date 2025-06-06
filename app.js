import { ChatGroq } from "@langchain/groq";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import * as dotenv from 'dotenv';
import { Document } from "@langchain/core/documents";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
//import { RecursiveCharaterTextSplitter } from "langchain/text_splitter";
//import { OllamaEmbeddings } from "@langchain/ollama";
//import {  MemoryVectorStore } from "langchain/chains/retrieval";

dotenv.config();

const llm = new ChatGroq({
    model: "meta-llama/llama-4-scout-17b-16e-instruct",
    temperature: 0.7,
})

const prompt = ChatPromptTemplate.fromTemplate(`
        Answer the user's question.
        Context: {context}
        Question: {input}
    `)

const resume_aurel = new Document({
    pageContent: "Aurel is a web developpeur, with skil in Symfony PHP, and linux"
});

const chain = await createStuffDocumentsChain({
    prompt,
    llm,
    documentSeparator: [resume_aurel]
})


const response = await chain.invoke({
    input: "What is Aurel's job?",
    context: [resume_aurel],
})


console.log(response);