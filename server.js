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
const { generateCommentSummary, createInteractiveSummarizerSession } = require('./summarizer');
const { createRagService } = require('./rag');
let createHuggingFaceProvider;
try {
  ({ createHuggingFaceProvider } = require('./providers/huggingface'));
} catch (e) {
  createHuggingFaceProvider = null;
}

const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '2mb' }));

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

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.post('/summarize', async (req, res) => {
  try {
    const { comments = [], filter = 'all', providerModel } = req.body || {};
    const provider = process.env.HF_API_KEY ? makeProvider(providerModel) : null;
    const { text, counts } = await generateCommentSummary(comments, filter, { provider });
    res.json({ summaryText: text, counts });
  } catch (err) {
    res.status(400).json({ error: String(err.message || err) });
  }
});

app.post('/details', async (req, res) => {
  try {
    const { comments = [], filter = 'all', providerModel } = req.body || {};
    const provider = process.env.HF_API_KEY ? makeProvider(providerModel) : null;
    const session = await createInteractiveSummarizerSession(comments, filter, { provider });
    res.json({ detailedSummaryText: session.generateMoreDetailedSummary() });
  } catch (err) {
    res.status(400).json({ error: String(err.message || err) });
  }
});

app.post('/compare', async (req, res) => {
  try {
    const { comments = [], filter = 'all', providerModel } = req.body || {};
    const provider = process.env.HF_API_KEY ? makeProvider(providerModel) : null;
    const session = await createInteractiveSummarizerSession(comments, filter, { provider });
    res.json({ comparisonText: session.comparePositiveVsNegative() });
  } catch (err) {
    res.status(400).json({ error: String(err.message || err) });
  }
});

app.post('/examples', async (req, res) => {
  try {
    const { comments = [], filter = 'all', keywords = [], sentiment, providerModel } = req.body || {};
    const provider = process.env.HF_API_KEY ? makeProvider(providerModel) : null;
    const session = await createInteractiveSummarizerSession(comments, filter, { provider });
    const examples = session.showExamplesFromComments(Array.isArray(keywords) ? keywords : [], sentiment);
    res.json({ examples });
  } catch (err) {
    res.status(400).json({ error: String(err.message || err) });
  }
});

app.post('/explain', async (req, res) => {
  try {
    const { comments = [], filter = 'all', pointIndex = 1, providerModel } = req.body || {};
    const provider = process.env.HF_API_KEY ? makeProvider(providerModel) : null;
    const session = await createInteractiveSummarizerSession(comments, filter, { provider });
    const details = session.explainPoint(Number(pointIndex) || 1);
    res.json(details);
  } catch (err) {
    res.status(400).json({ error: String(err.message || err) });
  }
});

app.post('/ask', async (req, res) => {
  try {
    const { comments = [], filter = 'all', question, providerModel } = req.body || {};
    if (!question || typeof question !== 'string') {
      return res.status(400).json({ error: 'question is required' });
    }
    const provider = process.env.HF_API_KEY ? makeProvider(providerModel) : null;
    const session = await createInteractiveSummarizerSession(comments, filter, { provider });
    const result = await session.ask(question);
    res.json(result);
  } catch (err) {
    res.status(400).json({ error: String(err.message || err) });
  }
});

app.post('/rag/chat', async (req, res) => {
  try {
    const {
      consultationId,
      comments = [],
      filter = 'all',
      question,
      intent = 'qa',
      sessionId
    } = req.body || {};
    if (!consultationId) {
      return res.status(400).json({ error: 'consultationId is required' });
    }
    if (!Array.isArray(comments)) {
      return res.status(400).json({ error: 'comments must be an array' });
    }
    if ((intent || 'qa') === 'qa' && (!question || typeof question !== 'string')) {
      return res.status(400).json({ error: 'question is required' });
    }

    const rag = getRagService();
    const result = await rag.chat({ consultationId, comments, filter, question, intent, sessionId });
    res.json(result);
  } catch (err) {
    res.status(400).json({ error: String(err.message || err) });
  }
});

const port = process.env.PORT || 8081;
if (require.main === module) {
  app.listen(port, () => {
    console.log(`Summarizer API listening on port ${port}`);
  });
}

module.exports = app;
