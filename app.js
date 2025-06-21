import { ChatGroq } from "@langchain/groq";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import * as dotenv from 'dotenv';
import { Document } from "@langchain/core/documents";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";


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

//Exemple charger un document PDF
const pdfLoader = new PDFLoader("test.pdf");
const pdfDocs = await pdfLoader.load();
//console.log(pdfDocs);

//charge le contenu de la page web
const loader = new CheerioWebBaseLoader("https://python.langchain.com/docs/integrations/document_loaders/spider/");
const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 250,
    chunkOverlap: 50,
});

//Découpage du contenu en morceaux de 250 caracteres
const splittedDocs = await splitter.splitDocuments(docs);
//console.log(splittedDocs)

// Ajoute le pdf au document splitted
splittedDocs.push(...pdfDocs);
console.log(splittedDocs)


// Met en place l'embeddings 
const ollamaEmbeddings = new OllamaEmbeddings({
    model: "llama3.2",
    baseUrl: "http://127.0.0.1:11434"
});

//Crée un vecteur de documents à partir des morceaux de documents
const vectorStore = await MemoryVectorStore.fromDocuments(splittedDocs, ollamaEmbeddings);

const retriever = vectorStore.asRetriever({
    k: 2,
});

const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever,
});


const response = await retrievalChain.invoke({
    input: "Qui est Aurel ?",
})


console.log(response); 