// RAG chat service for comment analysis using Supabase + embeddings.

const { createClient } = require('@supabase/supabase-js');
const { createHfEmbeddingsProvider } = require('./providers/hf-embeddings');
const { createOpenRouterProvider } = require('./providers/openrouter');

const STOPWORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'so', 'because', 'as', 'of', 'at', 'by', 'for', 'to',
  'from', 'in', 'on', 'with', 'without', 'about', 'into', 'over', 'under', 'between', 'is', 'are', 'was',
  'were', 'be', 'been', 'being', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
  'they', 'we', 'my', 'your', 'his', 'her', 'their', 'our', 'me', 'him', 'them', 'us', 'not', 'no', 'yes',
  'do', 'does', 'did', 'done', 'can', 'could', 'should', 'would', 'may', 'might', 'will', 'just', 'very',
  'really', 'more', 'most', 'less', 'least', 'have', 'has', 'had', 'having', 'also', 'too', 'than', 'such'
]);

const DEFAULT_TOP_K = 8;
const DEFAULT_SUMMARY_TOP_K = 12;
const MAX_CONTEXT_CHARS = 4000;
const EMBEDDING_BATCH_SIZE = 24;
const HISTORY_LIMIT = 6;
const MAX_EMBED_TEXT_CHARS = 1200;

let cachedSupabase;

function getSupabaseAdmin() {
  if (cachedSupabase) return cachedSupabase;
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) {
    throw new Error('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required');
  }
  cachedSupabase = createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
  return cachedSupabase;
}

function normalizeIntent(intent) {
  const raw = String(intent || 'qa').toLowerCase();
  if (raw === 'summary' || raw === 'summarize') return 'summary';
  if (raw === 'short-notes' || raw === 'shortnotes' || raw === 'notes') return 'short-notes';
  if (raw === 'feedback') return 'feedback';
  return 'qa';
}

function normalizeFilter(filter) {
  if (filter === 'positive only') return 'positive';
  if (filter === 'negative only') return 'negative';
  return null;
}

function normalizeSentiment(value) {
  if (value === 'positive' || value === 'negative' || value === 'neutral') return value;
  return 'neutral';
}

function sanitizeComments(comments) {
  const list = Array.isArray(comments) ? comments : [];
  return list
    .filter(c => c && typeof c.content === 'string' && c.content.trim().length > 0)
    .map(c => ({
      id: c.id || null,
      content: String(c.content).trim(),
      sentimenttype: normalizeSentiment(c.sentimenttype)
    }));
}

function chunkArray(items, size) {
  const out = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

function trimEmbedText(text) {
  const cleaned = String(text || '').trim();
  if (cleaned.length <= MAX_EMBED_TEXT_CHARS) return cleaned;
  return cleaned.slice(0, MAX_EMBED_TEXT_CHARS);
}

function tokenizeLite(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/g)
    .filter(t => t && t.length > 2 && !STOPWORDS.has(t));
}

function keywordRetrieve(comments, query, topK) {
  const tokens = tokenizeLite(query);
  if (!tokens.length) return comments.slice(0, topK);
  return comments
    .map(c => {
      const cTokens = tokenizeLite(c.content);
      const score = tokens.reduce((acc, t) => acc + (cTokens.includes(t) ? 1 : 0), 0);
      return { ...c, score };
    })
    .filter(c => c.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

async function fetchExistingMap(supabase, commentIds) {
  const existingMap = new Map();
  const batches = chunkArray(commentIds, 200);
  for (const batch of batches) {
    const { data, error } = await supabase
      .from('rag_comment_chunks')
      .select('comment_id, content')
      .in('comment_id', batch);
    if (error) {
      throw new Error(error.message);
    }
    (data || []).forEach(row => existingMap.set(row.comment_id, row.content));
  }
  return existingMap;
}

async function ensureEmbeddings({ supabase, embeddingsProvider, comments, consultationId }) {
  if (!embeddingsProvider) return { indexed: 0, skipped: comments.length };
  const commentIds = comments.map(c => c.id).filter(Boolean);
  if (!commentIds.length) return { indexed: 0, skipped: comments.length };

  const existingMap = await fetchExistingMap(supabase, commentIds);
  const toEmbed = comments.filter(c => c.id && c.content && existingMap.get(c.id) !== c.content);

  if (!toEmbed.length) return { indexed: 0, skipped: comments.length };

  let indexed = 0;
  for (const batch of chunkArray(toEmbed, EMBEDDING_BATCH_SIZE)) {
    const texts = batch.map(c => trimEmbedText(c.content));
    const vectors = await embeddingsProvider.embed(texts);
    if (!Array.isArray(vectors) || vectors.length !== batch.length) {
      throw new Error('Embedding response length mismatch');
    }
    const rows = batch.map((c, i) => ({
      comment_id: c.id,
      consultation_id: consultationId,
      content: c.content,
      sentimenttype: c.sentimenttype,
      embedding: vectors[i]
    }));
    const { error } = await supabase
      .from('rag_comment_chunks')
      .upsert(rows, { onConflict: 'comment_id' });
    if (error) throw new Error(error.message);
    indexed += rows.length;
  }

  return { indexed, skipped: comments.length - indexed };
}

async function retrieveMatches({ supabase, embeddingsProvider, comments, consultationId, query, filter, topK }) {
  if (embeddingsProvider) {
    const vectors = await embeddingsProvider.embed([query]);
    const embedding = vectors[0];
    const { data, error } = await supabase.rpc('match_comment_chunks', {
      query_embedding: embedding,
      match_count: topK,
      filter_consultation_id: consultationId,
      filter_sentiment: normalizeFilter(filter)
    });
    if (error) throw new Error(error.message);
    return (data || []).map(row => ({
      comment_id: row.comment_id,
      consultation_id: row.consultation_id,
      content: row.content,
      sentimenttype: row.sentimenttype,
      similarity: row.similarity
    }));
  }

  const filtered = normalizeFilter(filter)
    ? comments.filter(c => c.sentimenttype === normalizeFilter(filter))
    : comments;
  return keywordRetrieve(filtered, query, topK).map(c => ({
    comment_id: c.id,
    consultation_id: consultationId,
    content: c.content,
    sentimenttype: c.sentimenttype,
    similarity: null
  }));
}

function buildSystemPrompt(intent) {
  const base = [
    'You are an assistant for analyzing user comment feedback.',
    'Use ONLY the provided comment context. If the answer is not in the context, say you do not have enough info.',
    'Keep the response concise and professional.'
  ];
  const intentGuide = {
    summary: 'Provide a 4-6 sentence summary and then 3-5 bullet key points.',
    'short-notes': 'Provide short notes as 5-8 bullet points.',
    feedback: 'Provide actionable feedback and improvement ideas in bullet points.',
    qa: 'Answer the user question directly in 3-6 sentences.'
  };
  return [...base, intentGuide[intent] || intentGuide.qa].join(' ');
}

function buildUserPrompt(intent, question) {
  if (intent === 'summary') return 'Summarize the overall feedback in the comments.';
  if (intent === 'short-notes') return 'Create short notes from the comments.';
  if (intent === 'feedback') return 'Provide feedback and improvement ideas based on the comments.';
  return String(question || '').trim();
}

function buildContext(matches, filter) {
  let total = 0;
  const lines = [];
  for (let i = 0; i < matches.length; i++) {
    const m = matches[i];
    const line = `[${i + 1}] (${m.sentimenttype || 'neutral'}) ${m.content}`;
    if (total + line.length > MAX_CONTEXT_CHARS) break;
    lines.push(line);
    total += line.length + 1;
  }
  const filterLabel = filter && filter !== 'all' ? `Filter: ${filter}` : 'Filter: all';
  return [filterLabel, 'Context:', ...lines].join('\n');
}

async function ensureSession(supabase, sessionId, consultationId) {
  if (sessionId) {
    const { data, error } = await supabase
      .from('rag_chat_sessions')
      .select('id, consultation_id')
      .eq('id', sessionId)
      .maybeSingle();
    if (error) throw new Error(error.message);
    if (data && data.consultation_id === consultationId) return data.id;
  }
  const { data, error } = await supabase
    .from('rag_chat_sessions')
    .insert({ consultation_id: consultationId })
    .select('id')
    .single();
  if (error) throw new Error(error.message);
  return data.id;
}

async function loadHistory(supabase, sessionId, limit) {
  const { data, error } = await supabase
    .from('rag_chat_messages')
    .select('role, content, created_at')
    .eq('session_id', sessionId)
    .order('created_at', { ascending: false })
    .limit(limit);
  if (error) throw new Error(error.message);
  return (data || []).reverse().map(row => ({ role: row.role, content: row.content }));
}

async function appendMessages(supabase, sessionId, messages) {
  if (!messages.length) return;
  const rows = messages.map(m => ({
    session_id: sessionId,
    role: m.role,
    content: m.content
  }));
  const { error } = await supabase.from('rag_chat_messages').insert(rows);
  if (error) throw new Error(error.message);
}

function createRagService(options = {}) {
  const supabase = options.supabase || getSupabaseAdmin();
  const embeddingsProvider = options.embeddingsProvider ||
    (process.env.HF_API_KEY
      ? createHfEmbeddingsProvider({ model: options.embeddingModel || process.env.HF_EMBED_MODEL })
      : null);
  const chatProvider = options.chatProvider || createOpenRouterProvider({ model: options.chatModel });

  async function chat({
    consultationId,
    comments,
    filter = 'all',
    question,
    intent = 'qa',
    sessionId
  }) {
    if (!consultationId) throw new Error('consultationId is required');
    const cleanedComments = sanitizeComments(comments);
    if (!cleanedComments.length) {
      return { sessionId: null, answer: 'No comments are available for this consultation.', sources: [] };
    }

    const normalizedIntent = normalizeIntent(intent);
    const userPrompt = buildUserPrompt(normalizedIntent, question);
    if (normalizedIntent === 'qa' && !userPrompt) {
      throw new Error('question is required');
    }

    let activeEmbeddingsProvider = embeddingsProvider;
    try {
      await ensureEmbeddings({
        supabase,
        embeddingsProvider,
        comments: cleanedComments,
        consultationId
      });
    } catch (err) {
      activeEmbeddingsProvider = null;
    }

    const topK = normalizedIntent === 'qa' ? DEFAULT_TOP_K : DEFAULT_SUMMARY_TOP_K;
    let matches;
    try {
      matches = await retrieveMatches({
        supabase,
        embeddingsProvider: activeEmbeddingsProvider,
        comments: cleanedComments,
        consultationId,
        query: userPrompt || 'overall feedback summary',
        filter,
        topK
      });
    } catch (err) {
      const filtered = normalizeFilter(filter)
        ? cleanedComments.filter(c => c.sentimenttype === normalizeFilter(filter))
        : cleanedComments;
      matches = filtered.slice(0, topK).map(c => ({
        comment_id: c.id,
        consultation_id: consultationId,
        content: c.content,
        sentimenttype: c.sentimenttype,
        similarity: null
      }));
    }

    if (!matches || matches.length === 0) {
      const sentimentFilter = normalizeFilter(filter);
      const base = sentimentFilter
        ? cleanedComments.filter(c => c.sentimenttype === sentimentFilter)
        : cleanedComments;
      matches = base.slice(0, topK).map(c => ({
        comment_id: c.id,
        consultation_id: consultationId,
        content: c.content,
        sentimenttype: c.sentimenttype,
        similarity: null
      }));
    }

    const contextText = buildContext(matches, filter);
    const systemPrompt = buildSystemPrompt(normalizedIntent);
    const activeSessionId = await ensureSession(supabase, sessionId, consultationId);
    const history = await loadHistory(supabase, activeSessionId, HISTORY_LIMIT);

    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'system', content: contextText },
      ...history,
      { role: 'user', content: userPrompt }
    ];

    const answer = await chatProvider.chat(messages, { temperature: 0.4, max_tokens: 700 });

    await appendMessages(supabase, activeSessionId, [
      { role: 'user', content: userPrompt },
      { role: 'assistant', content: answer }
    ]);

    return {
      sessionId: activeSessionId,
      answer,
      intent: normalizedIntent,
      sources: matches.map(m => ({
        comment_id: m.comment_id,
        content: m.content,
        sentimenttype: m.sentimenttype,
        similarity: m.similarity
      }))
    };
  }

  return { chat };
}

module.exports = { createRagService };
