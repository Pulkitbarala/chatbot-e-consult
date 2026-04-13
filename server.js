// Express API for comment summarization, designed for Render deployment.
// Endpoints:
// - POST /summarize { comments, filter, providerModel? }
// - POST /details { comments, filter }
// - POST /compare { comments, filter }
// - POST /examples { comments, filter, keywords, sentiment? }
// - POST /explain { comments, filter, pointIndex }
// - POST /ask { comments, filter, question }

const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '.env') });

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const rateLimit = require('express-rate-limit');
const { z } = require('zod');
const { generateCommentSummary, createInteractiveSummarizerSession } = require('./summarizer');
const { createRagService } = require('./rag');
let createHuggingFaceProvider;
try {
  ({ createHuggingFaceProvider } = require('./providers/huggingface'));
} catch (e) {
  createHuggingFaceProvider = null;
}

function toInt(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

const MAX_COMMENTS = toInt(process.env.RAG_MAX_COMMENTS, 500);
const MAX_COMMENT_CHARS = toInt(process.env.RAG_MAX_COMMENT_CHARS, 4000);

const allowedOrigins = (process.env.CORS_ORIGINS || [
  'http://localhost:5173',
  'http://127.0.0.1:5173',
  'https://e-consult-zeta.vercel.app',
  'https://e-consult-backend.vercel.app',
  'https://chatbot-e-consult.vercel.app/rag/chat'
].join(',')).split(',').map(s => s.trim()).filter(Boolean);

const corsOptions = {
  origin: (origin, callback) => {
    if (!origin) return callback(null, true);
    if (allowedOrigins.includes(origin)) return callback(null, true);
    return callback(null, false);
  },
  credentials: true,
};

const limiter = rateLimit({
  windowMs: toInt(process.env.RATE_LIMIT_WINDOW_MS, 60_000),
  max: toInt(process.env.RATE_LIMIT_MAX, 60),
  standardHeaders: true,
  legacyHeaders: false,
});

function requireApiKey(req, res, next) {
  const expected = process.env.SUMMARIZER_API_KEY;
  if (!expected) return next();

  const headerKey = req.headers['x-api-key'] || req.headers['x-summarizer-key'];
  const authHeader = req.headers['authorization'];
  const bearer = typeof authHeader === 'string' && authHeader.startsWith('Bearer ')
    ? authHeader.slice(7)
    : null;
  const provided = (Array.isArray(headerKey) ? headerKey[0] : headerKey) || bearer;

  if (!provided || String(provided).trim() !== String(expected).trim()) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  return next();
}

function sendError(res, err, status = 400) {
  const message = err?.message || String(err || 'Request failed');
  if (status >= 500) {
    return res.status(status).json({ error: 'Request failed' });
  }
  return res.status(status).json({ error: message || 'Request failed' });
}

const app = express();
app.use(cors(corsOptions));
app.use(limiter);
app.use(bodyParser.json({ limit: '8mb' }));

app.use((req, res, next) => {
  if (req.path === '/health') return next();
  return requireApiKey(req, res, next);
});

let ragService;
function getRagService() {
  if (!ragService) {
    ragService = createRagService();
  }
  return ragService;
}

function makeProvider(model) {
  if (!createHuggingFaceProvider) return null;
  try {
    return createHuggingFaceProvider(model ? { model } : {});
  } catch {
    return null;
  }
}

function sanitizeCommentsPayload(comments) {
  const list = Array.isArray(comments) ? comments : [];
  return list.slice(0, MAX_COMMENTS).map((c) => {
    const content = String(c?.content || '').slice(0, MAX_COMMENT_CHARS);
    return {
      id: c?.id,
      content,
      sentimenttype: c?.sentimenttype,
    };
  });
}

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

const CommentSchema = z.object({
  id: z.union([z.string(), z.number()]).optional(),
  content: z.string().min(1).max(MAX_COMMENT_CHARS),
  sentimenttype: z.enum(['positive', 'negative', 'neutral']).optional(),
});

const RagChatSchema = z.object({
  consultationId: z.string().min(1),
  comments: z.array(CommentSchema).max(MAX_COMMENTS),
  filter: z.enum(['positive only', 'negative only', 'all']).optional(),
  question: z.string().max(2000).optional(),
  intent: z.enum(['qa', 'summary', 'short-notes', 'feedback']).optional(),
  sessionId: z.string().optional(),
});

app.post('/summarize', async (req, res) => {
  try {
    const { comments = [], filter = 'all', providerModel } = req.body || {};
    const safeComments = sanitizeCommentsPayload(comments);
    const provider = process.env.HF_API_KEY ? makeProvider(providerModel) : null;
    const { text, counts } = await generateCommentSummary(safeComments, filter, { provider });
    res.json({ summaryText: text, counts });
  } catch (err) {
    sendError(res, err, 400);
  }
});

app.post('/details', async (req, res) => {
  try {
    const { comments = [], filter = 'all', providerModel } = req.body || {};
    const safeComments = sanitizeCommentsPayload(comments);
    const provider = process.env.HF_API_KEY ? makeProvider(providerModel) : null;
    const session = await createInteractiveSummarizerSession(safeComments, filter, { provider });
    res.json({ detailedSummaryText: session.generateMoreDetailedSummary() });
  } catch (err) {
    sendError(res, err, 400);
  }
});

app.post('/compare', async (req, res) => {
  try {
    const { comments = [], filter = 'all', providerModel } = req.body || {};
    const safeComments = sanitizeCommentsPayload(comments);
    const provider = process.env.HF_API_KEY ? makeProvider(providerModel) : null;
    const session = await createInteractiveSummarizerSession(safeComments, filter, { provider });
    res.json({ comparisonText: session.comparePositiveVsNegative() });
  } catch (err) {
    sendError(res, err, 400);
  }
});

app.post('/examples', async (req, res) => {
  try {
    const { comments = [], filter = 'all', keywords = [], sentiment, providerModel } = req.body || {};
    const safeComments = sanitizeCommentsPayload(comments);
    const provider = process.env.HF_API_KEY ? makeProvider(providerModel) : null;
    const session = await createInteractiveSummarizerSession(safeComments, filter, { provider });
    const examples = session.showExamplesFromComments(Array.isArray(keywords) ? keywords : [], sentiment);
    res.json({ examples });
  } catch (err) {
    sendError(res, err, 400);
  }
});

app.post('/explain', async (req, res) => {
  try {
    const { comments = [], filter = 'all', pointIndex = 1, providerModel } = req.body || {};
    const safeComments = sanitizeCommentsPayload(comments);
    const provider = process.env.HF_API_KEY ? makeProvider(providerModel) : null;
    const session = await createInteractiveSummarizerSession(safeComments, filter, { provider });
    const details = session.explainPoint(Number(pointIndex) || 1);
    res.json(details);
  } catch (err) {
    sendError(res, err, 400);
  }
});

app.post('/ask', async (req, res) => {
  try {
    const { comments = [], filter = 'all', question, providerModel } = req.body || {};
    if (!question || typeof question !== 'string') {
      return res.status(400).json({ error: 'question is required' });
    }
    const safeComments = sanitizeCommentsPayload(comments);
    const provider = process.env.HF_API_KEY ? makeProvider(providerModel) : null;
    const session = await createInteractiveSummarizerSession(safeComments, filter, { provider });
    const result = await session.ask(question);
    res.json(result);
  } catch (err) {
    sendError(res, err, 400);
  }
});

app.post('/rag/chat', async (req, res) => {
  try {
    const parsed = RagChatSchema.safeParse(req.body || {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid request' });
    }
    const { consultationId, comments, filter, question, intent, sessionId } = parsed.data;
    const rag = getRagService();
    const result = await rag.chat({ consultationId, comments, filter, question, intent, sessionId });
    res.json(result);
  } catch (err) {
    sendError(res, err, 400);
  }
});

const port = process.env.PORT || 8081;
if (require.main === module) {
  app.listen(port, () => {
    console.log(`Summarizer API listening on port ${port}`);
  });
}

module.exports = app;
