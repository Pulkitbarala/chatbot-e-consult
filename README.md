# Comment Summarizer

Summarize user comments by sentiment with optional LLM support (Hugging Face Inference API). Provides a strict output format and interactive follow-ups for dashboards.

## Features

- Filtering: `positive only`, `negative only`, or `all`
- Strict structured output (SUMMARY, KEY POINTS, SENTIMENT OVERVIEW)
- Optional LLM provider (e.g., Llama 3 via Hugging Face)
- Heuristic fallback (no dependencies) if provider unavailable
- Interactive APIs: explain points, show examples, more detailed summary, compare sentiments
- RAG chatbot with Supabase vector search + OpenRouter chat

## Usage (Code)

Import and call from your dashboard logic:

```js
const { generateCommentSummary, createInteractiveSummarizerSession } = require('./sumarizer/summarizer');
const { createHuggingFaceProvider } = require('./sumarizer/providers/huggingface');

// comments: [{ text: string, sentiment: 'positive'|'negative'|'neutral' }]
async function summarize(comments, filter) {
  const provider = process.env.HF_API_KEY ? createHuggingFaceProvider() : null;
  const { text } = await generateCommentSummary(comments, filter, { provider });
  return text; // Render directly in the UI
}

async function interactive(comments, filter) {
  const provider = process.env.HF_API_KEY ? createHuggingFaceProvider() : null;
  const session = await createInteractiveSummarizerSession(comments, filter, { provider });
  console.log(session.summaryText);
  console.log(session.generateMoreDetailedSummary());
  console.log(session.comparePositiveVsNegative());
  console.log(session.explainPoint(1));
}
```

## Hugging Face Setup (Optional)

1. Create a token at https://huggingface.co/settings/tokens
2. Set `HF_API_KEY` in your environment.

### Windows PowerShell

```powershell
$env:HF_API_KEY = "hf_xxx_your_token"
```

By default, the provider uses `meta-llama/Llama-3.1-8B-Instruct`. You can change it:

```js
const provider = createHuggingFaceProvider({ model: 'google/gemma-2-9b-it' });
```

## RAG Setup (Supabase + OpenRouter)

The RAG chatbot stores embeddings and chat history in Supabase, and uses OpenRouter for chat completion.

Required environment variables:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `OPENROUTER_API_KEY`

Optional environment variables:

- `OPENROUTER_MODEL` (default: `openrouter/auto`)
- `OPENROUTER_APP_URL` and `OPENROUTER_APP_NAME` (optional headers)
- `HF_API_KEY` (required for embeddings)
- `HF_EMBED_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)

## Run Demo (Local)

```powershell
node sumarizer/runner.js all
```

Use `positive only` or `negative only` in place of `all` to filter.

## API Service (Render)

Deploy the API in `sumarizer` as a Render web service:

- Root Directory: `sumarizer`
- Build Command: `npm install`
- Start Command: `npm start`
- Environment: add `HF_API_KEY` (optional)

### Endpoints

- POST `/summarize`
  - Body: `{ comments: [{ text, sentiment }], filter: 'positive only'|'negative only'|'all', providerModel?: string }`
  - Returns: `{ summaryText, counts }`

- POST `/details`
  - Body: `{ comments, filter }`
  - Returns: `{ detailedSummaryText }`

- POST `/compare`
  - Body: `{ comments, filter }`
  - Returns: `{ comparisonText }`

- POST `/examples`
  - Body: `{ comments, filter, keywords: string[], sentiment?: 'positive'|'negative'|'neutral' }`
  - Returns: `{ examples: [{ sentiment, text }] }`

- POST `/explain`
  - Body: `{ comments, filter, pointIndex: number }`
  - Returns: `{ keywords: string[], examples: [{ sentiment, text }] }`

- POST `/ask`
  - Body: `{ comments, filter, question: string }`
  - Returns: `{ answer: string, examples: [{ sentiment, text }] }`

- POST `/rag/chat`
  - Body: `{ consultationId: string, comments: [{ id, content, sentimenttype }], filter: 'positive only'|'negative only'|'all', intent?: 'qa'|'summary'|'short-notes'|'feedback', question?: string, sessionId?: string }`
  - Returns: `{ sessionId: string, answer: string, intent: string, sources: [{ comment_id, content, sentimenttype, similarity }] }`

## Frontend Integration (React)

Call the API from your dashboard when the user enters AI mode:

```ts
// Example: src/pages/AnalysisDashboard.tsx (pseudo-code)
async function summarize(comments, filter) {
  const res = await fetch(process.env.VITE_SUMMARIZER_URL + '/summarize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ comments, filter }),
  });
  return res.json(); // { summaryText, counts }
}

async function details(comments, filter) {
  const res = await fetch(process.env.VITE_SUMMARIZER_URL + '/details', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ comments, filter }),
  });
  return res.json();
}

async function compare(comments, filter) {
  const res = await fetch(process.env.VITE_SUMMARIZER_URL + '/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ comments, filter }),
  });
  return res.json(); // { comparisonText }
}

async function examples(comments, filter, keywords, sentiment) {
  const res = await fetch(process.env.VITE_SUMMARIZER_URL + '/examples', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ comments, filter, keywords, sentiment }),
  });
  return res.json(); // { examples }
}

async function explain(comments, filter, pointIndex) {
  const res = await fetch(process.env.VITE_SUMMARIZER_URL + '/explain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ comments, filter, pointIndex }),
  });
  return res.json(); // { keywords, examples }
}

async function ask(comments, filter, question) {
  const res = await fetch(process.env.VITE_SUMMARIZER_URL + '/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ comments, filter, question }),
  });
  return res.json(); // { answer, examples }
}

async function ragChat(consultationId, comments, filter, question, intent, sessionId) {
  const res = await fetch(process.env.VITE_SUMMARIZER_URL + '/rag/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ consultationId, comments, filter, question, intent, sessionId }),
  });
  return res.json(); // { sessionId, answer, sources }
}

// Use buttons like "Enter AI Mode" to trigger these calls.
```

Set `VITE_SUMMARIZER_URL` in your frontend environment to the Render service URL.

### Local Offline Mode

- The frontend now falls back to `http://localhost:8080` when `VITE_SUMMARIZER_URL` is not set.
- Start the API locally:

```powershell
cd sumarizer
npm install
npm start
```

- Then run the frontend dev server (in project root):

```powershell
npm run dev
```

- Open your dashboard; “Enter AI Mode” will call the local API.
If you change server code, restart it to pick up changes:

```powershell
# In the sumarizer folder
Ctrl+C            # stop the running server
npm start         # start again
```

## Output Structure

The summarizer returns a string formatted as:

```
SUMMARY:
<4-6 sentences>

KEY POINTS:
* point 1
* point 2
* point 3

SENTIMENT OVERVIEW:
Positive comments: <count>
Negative comments: <count>
Neutral comments: <count>
```

## Notes

- If the LLM provider fails or is not configured, the heuristic summarizer produces concise, professional summaries.
- Interactive methods operate over the filtered set to keep follow-ups consistent with the user’s selection.
 - Large datasets (up to ~10k comments) are handled automatically via map-reduce chunking; tune `chunkSize` and `chunkThreshold` on the server if needed.
