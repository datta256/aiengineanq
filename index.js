require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const fetch = require('node-fetch');
const { Ollama } = require('@langchain/community/llms/ollama');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { PromptTemplate } = require('@langchain/core/prompts');
const { LLMChain } = require('langchain/chains');
const { OllamaEmbeddings } = require('@langchain/community/embeddings/ollama');

const app = express();
app.use(express.json());
app.use(cors());

/** ---------------- Config ---------------- */
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://10.5.20.209:11434';
const CLASSIFIER_MODEL = process.env.CLASSIFIER_MODEL || 'mistral:latest';
const CHAT_MODEL = process.env.CHAT_MODEL || 'mistral:latest';
const EMBED_MODEL = process.env.EMBED_MODEL || 'nomic-embed-text';
const KNOWLEDGE_FILES = (process.env.KNOWLEDGE_FILES || 'knowledge.txt')
  .split(',')
  .map(s => s.trim())
  .filter(Boolean);

const embeddings = new OllamaEmbeddings({
  baseUrl: "http://127.0.0.1:11434",
  model: EMBED_MODEL,
});

/** ---------------- Cosine Similarity ---------------- */
function cosineSimilarity(vecA, vecB) {
  if (vecA.length !== vecB.length) throw new Error("Vectors must have the same length");
  const dot = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dot / (magA * magB);
}

/** ---------------- Role Classification ---------------- */

const agentPrompts = {
  customer_support: "You are a helpful and empathetic customer support agent. ONLY answer from the given context. If the answer is not found, say: 'will forward this to our customer agent'.",
  sales_agent: "You are a persuasive and friendly sales representative. ONLY answer from the given context. If the answer is not found, say: 'will forward this to our sales agent'.",
  marketing_agent: "You are a creative and data-driven marketing strategist. ONLY answer from the given context. If the answer is not found, say: 'will forward this to our marketing agent'.",
  technical_expert: "You are a precise and experienced technical expert. ONLY answer from the given context. If the answer is not found, say: 'will forward this to our technical expert'.",
  general_info: "You are a polite and intelligent assistant helping with general queries. ONLY answer from the given context. If the answer is not found, say: 'will forward this to our general info agent'.",
};

const roleClassifierPrompt = PromptTemplate.fromTemplate(`
You are a role classifier for an AI assistant.

Given a user query, classify the best suited agent role to handle it:
- customer_support
- sales_agent
- marketing_agent
- technical_expert
- general_info

Respond ONLY with the role.

Examples:
User: My order hasn’t arrived.
Role: customer_support

User: What are your pricing plans?
Role: sales_agent

User: How do I use the API?
Role: technical_expert

User: Help me write a product launch announcement.
Role: marketing_agent

User: What’s the capital of Japan?
Role: general_info

User: {input}
Role:
`);

const roleChain = new LLMChain({
  llm: new Ollama({
    baseUrl: OLLAMA_BASE_URL,
    model: CLASSIFIER_MODEL,
    temperature: 0,
    maxTokens: 50,
  }),
  prompt: roleClassifierPrompt,
});

async function classifyRole(query) {
  const result = await roleChain.call({ input: query });
  const label = result.text.trim().toLowerCase();
  const allowed = Object.keys(agentPrompts);
  return allowed.includes(label) ? label : 'general_info';
}

/** ---------------- Question Validation ---------------- */

const questionValidationPrompt = PromptTemplate.fromTemplate(`
You are a question validation assistant.

Determine if the user's query is valid and has enough context to answer.

Respond with ONLY one of:
- valid
- needs_more_context

Examples:
User: tx blocked
Answer: needs_more_context

User: How do I reset my password?
Answer: valid

User: error?
Answer: needs_more_context

User: {input}
Answer:
`);

const validationChain = new LLMChain({
  llm: new Ollama({
    baseUrl: OLLAMA_BASE_URL,
    model: CLASSIFIER_MODEL,
    temperature: 0,
    maxTokens: 10,
  }),
  prompt: questionValidationPrompt,
});

async function validateQuestion(query) {
  const result = await validationChain.call({ input: query });
  const label = result.text.trim().toLowerCase();
  return label === "valid";
}

/** ---------------- Knowledge Base + Embeddings ---------------- */

let vectorStore = []; // [{ embedding, text, fileName }]

async function loadKnowledgeBase(filePaths = KNOWLEDGE_FILES) {
  if (vectorStore.length > 0) return vectorStore;

  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 100 });

  for (const path of filePaths) {
    if (!fs.existsSync(path)) continue;
    const text = fs.readFileSync(path, 'utf8').trim();
    if (!text) continue;

    const docs = await splitter.createDocuments([text]);

    for (const doc of docs) {
      const embedding = await embeddings.embedQuery(doc.pageContent);
      vectorStore.push({
        embedding,
        text: doc.pageContent,
        fileName: path,
      });
    }
  }

  return vectorStore;
}

async function findRelevantChunks(question, topK = 3, threshold = 0.3) {
  const queryEmbedding = await embeddings.embedQuery(question);

  const scored = vectorStore.map(item => ({
    ...item,
    score: cosineSimilarity(queryEmbedding, item.embedding),
  }));

  const sorted = scored.sort((a, b) => b.score - a.score);
  const topResults = sorted.slice(0, topK);

  console.log(`Top ${topK} results for query "${question}":`, topResults.map(r => r.fileName));

  if (topResults[0]?.score < threshold) {
    return []; // Not confident enough → return empty to prevent hallucination
  }

  return topResults;
}

/** ---------------- Ask Route ---------------- */

app.post('/ask', async (req, res) => {
  const { query } = req.body;
  if (!query) return res.status(400).json({ error: 'Missing query.' });

  // Step 1: Validate question
  const isValid = await validateQuestion(query);
  if (!isValid) {
    return res.send("Could you provide more details about your question?");
  }

  try {
    // Step 2: Classify role
    const role = await classifyRole(query);
    const basePrompt = agentPrompts[role] || agentPrompts.general_info;

    // Step 3: Load knowledge base and find relevant chunks
    await loadKnowledgeBase();
    const relevantChunks = await findRelevantChunks(query, 3);

    if (relevantChunks.length === 0) {
      return res.send(`will forward this to our ${role.replace('_', ' ')}`);
    }

    const context = relevantChunks.map(({ text, fileName }) =>
      `From ${fileName}:\n${text.slice(0, 600)}`
    ).join('\n\n');

    // Step 4: Create system prompt
    const systemPrompt = `
${basePrompt}

RULES:
- ONLY answer using the Reference Context.
- If answer not found in context reply exactly: "will forward this to our ${role.replace('_', ' ')}".

Reference Context:
${context}
`;

    // Step 5: Ask the model
    const response = await fetch(`${OLLAMA_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: CHAT_MODEL,
        stream: false,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: query }
        ],
        options: {
          temperature: 1,
        }
      }),
    });

    const data = await response.json();
    const fullAnswer = data.message?.content || '❌ No content returned.';

    res.send(fullAnswer);

  } catch (err) {
    console.error(err);
    res.status(500).send('❌ An error occurred.');
  }
});

/** ---------------- Health Check ---------------- */
app.get('/', (req, res) => res.send('✅ AI Agent Server is running!'));

/** ---------------- Start Server ---------------- */
const PORT = process.env.PORT || 3001;
app.listen(PORT, '0.0.0.0', () =>
  console.log(`🧠 AI Agent Server running on http://0.0.0.0:${PORT}`)
);
