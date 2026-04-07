// Simple, dependency-free comment summarizer with optional LLM provider.
// Works standalone, and exposes helper APIs for interactive follow-ups.

/**
 * @typedef {{ id?: string|number, text: string, sentiment: 'positive'|'negative'|'neutral' }} Comment
 * @typedef {'positive only'|'negative only'|'all'} FilterOption
 * @typedef {{
 *   provider?: {
 *     name: string,
 *     summarize: (prompt: string) => Promise<string>
 *   },
 *   maxExamplesPerPoint?: number,
 *   outputControls?: {
 *     summarySentenceCount?: number,
 *     keyPointsCount?: number
 *   },
 *   mapReduce?: boolean,
 *   chunkSize?: number,
 *   chunkThreshold?: number
 * }} SummarizerOptions
 */

const STOPWORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'so', 'because', 'as', 'of', 'at', 'by', 'for', 'to', 'from', 'in', 'on', 'with', 'without', 'about', 'into', 'over', 'under', 'between', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'they', 'we', 'my', 'your', 'his', 'her', 'their', 'our', 'me', 'him', 'them', 'us', 'not', 'no', 'yes', 'do', 'does', 'did', 'done', 'can', 'could', 'should', 'would', 'may', 'might', 'will', 'just', 'very', 'really', 'more', 'most', 'less', 'least', 'have', 'has', 'had', 'having', 'also', 'too', 'than', 'such', 'there', 'here', 'out', 'up', 'down'
]);

const SENTIMENT_WORDS = new Set([
  'good', 'bad', 'great', 'excellent', 'poor', 'awesome', 'terrible', 'nice', 'love', 'hate', 'worst', 'best', 'amazing', 'awful'
]);

const FILLER_WORDS = new Set([
  'think', 'quite', 'now', 'user', 'users', 'comment', 'comments', 'thing', 'things', 'talk', 'talking', 'say', 'said', 'post', 'posts', 'like'
]);

function normalizeText(text) {
  return String(text)
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function tokenize(text) {
  return normalizeText(text)
    .split(' ')
    .filter(t => t && !STOPWORDS.has(t) && t.length > 2);
}

function chunkArray(arr, size) {
  const out = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

function countSentiments(comments) {
  return comments.reduce((acc, c) => {
    acc[c.sentiment] = (acc[c.sentiment] || 0) + 1;
    return acc;
  }, { positive: 0, negative: 0, neutral: 0 });
}

/**
 * @param {Comment[]} comments
 * @param {FilterOption} filter
 * @returns {{ filtered: Comment[], counts: { positive: number, negative: number, neutral: number } }}
 */
function filterComments(comments, filter) {
  const all = Array.isArray(comments) ? comments : [];
  const filtered = all.filter(c => {
    if (filter === 'positive only') return c.sentiment === 'positive';
    if (filter === 'negative only') return c.sentiment === 'negative';
    return true;
  });
  const counts = countSentiments(filtered);
  return { filtered, counts };
}

/**
 * @param {Comment[]} comments
 * @param {{ minWordFreq?: number, minBigramFreq?: number }} [options]
 */
function buildStats(comments, options = {}) {
  const freq = Object.create(null);
  const bigrams = Object.create(null);
  for (const c of comments) {
    const toks = tokenize(c.text);
    for (let i = 0; i < toks.length; i++) {
      const t = toks[i];
      freq[t] = (freq[t] || 0) + 1;
      if (i < toks.length - 1) {
        const bi = `${t} ${toks[i + 1]}`;
        bigrams[bi] = (bigrams[bi] || 0) + 1;
      }
    }
  }

  const minWordFreq = Number.isFinite(options.minWordFreq) ? options.minWordFreq : 2;
  const minBigramFreq = Number.isFinite(options.minBigramFreq) ? options.minBigramFreq : 2;
  const sortedWords = Object.entries(freq).sort((a, b) => b[1] - a[1]);
  const sortedBigrams = Object.entries(bigrams).sort((a, b) => b[1] - a[1]);
  const topWords = sortedWords.filter(([_, n]) => n >= minWordFreq).slice(0, 10).map(([w]) => w);
  const topBigrams = sortedBigrams.filter(([_, n]) => n >= minBigramFreq).slice(0, 5).map(([w]) => w);
  const fallbackWords = topWords.length ? topWords : sortedWords.slice(0, 10).map(([w]) => w);
  const fallbackBigrams = topBigrams.length ? topBigrams : sortedBigrams.slice(0, 5).map(([w]) => w);
  return { topWords: fallbackWords, topBigrams: fallbackBigrams };
}

function filterKeywords(words) {
  return words.filter(w => w && !STOPWORDS.has(w) && !SENTIMENT_WORDS.has(w) && !FILLER_WORDS.has(w) && w.length > 2);
}

/**
 * Heuristic summary builder used when no provider is configured.
 * @param {Comment[]} comments
 * @param {{ positive: number, negative: number, neutral: number }} counts
 * @param {FilterOption} filter
 */
function buildHeuristicSummary(comments, counts, filter, options = {}) {
  const total = comments.length;
  if (total === 0) {
    return {
      summary: 'No comments available for the selected filter. Please adjust the filter or try again.',
      keyPoints: [
        'No data to analyze',
        'Try selecting a different sentiment filter',
        'Await more user feedback to summarize'
      ],
    };
  }

  const positives = comments.filter(c => c.sentiment === 'positive');
  const negatives = comments.filter(c => c.sentiment === 'negative');
  const neutrals = comments.filter(c => c.sentiment === 'neutral');
  const overallStats = buildStats(comments);
  const negStats = buildStats(negatives);
  const posStats = buildStats(positives);

  const themes = filterKeywords(overallStats.topWords).slice(0, 5);
  const concernsRaw = negatives.length > 0 ? filterKeywords(negStats.topWords).slice(0, 5) : [];
  const positivesRaw = positives.length > 0 ? filterKeywords(posStats.topWords).slice(0, 5) : [];
  const concerns = concernsRaw.length ? concernsRaw : themes;
  const positivesTop = positivesRaw.length ? positivesRaw : themes;
  const neutralWords = neutrals.length > 0 ? filterKeywords(buildStats(neutrals).topWords).slice(0, 3) : [];

  const sentimentLean = counts.positive > counts.negative
    ? 'leans positive'
    : (counts.negative > counts.positive ? 'leans negative' : 'appears balanced');
  const filterNote = filter === 'all' ? 'across all feedback' : `within ${filter}`;
  const bigramText = overallStats.topBigrams.slice(0, 2).join(' and ');

  const sentence1 = `Overall feedback ${sentimentLean} ${filterNote}.`;
  const sentence2 = negatives.length > 0
    ? (concerns.length ? `Main concerns center around ${concerns.slice(0, 3).join(', ')}.` : 'Main concerns vary across topics.')
    : (positivesTop.length ? `Few pressing concerns are evident; users highlight ${positivesTop.slice(0, 3).join(', ')}.` : 'Few pressing concerns are evident.');
  const sentence3 = themes.length ? `Common themes include ${themes.join(', ')}.` : 'Common themes are limited in this selection.';
  const sentence4 = neutralWords.length
    ? `Neutral comments often mention ${neutralWords.join(', ')} without strong sentiment.`
    : (neutrals.length > 0 ? 'Neutral comments provide context without strong sentiment.' : 'Neutral perspectives are limited in this selection.');
  const sentence5 = positivesTop.length
    ? `Positive feedback frequently praises ${positivesTop.slice(0, 3).join(', ')}.`
    : (positives.length > 0 ? 'Positive feedback references various aspects without a dominant theme.' : 'Positive feedback is limited in this selection.');
  const sentence6 = bigramText ? `Users repeatedly reference ${bigramText}.` : '';

  const summarySentences = [sentence1, sentence2, sentence3, sentence4, sentence5, sentence6].filter(Boolean);
  const requestedCount = options.outputControls?.summarySentenceCount;
  const maxSummarySentences = Number.isFinite(requestedCount)
    ? Math.min(Math.max(Math.floor(requestedCount), 2), summarySentences.length)
    : summarySentences.length;
  const summary = summarySentences.slice(0, maxSummarySentences).join(' ');

  const freqMentions = filterKeywords(overallStats.topWords).slice(0, 3);
  const basePoints = [
    negatives.length > 0
      ? (concerns.length ? `Concerns: ${concerns.slice(0, 3).join(', ')}` : 'Concerns: varied across topics')
      : (positivesTop.length ? `Strengths: ${positivesTop.slice(0, 3).join(', ')}` : 'Strengths: diverse without a dominant theme'),
    themes.length ? `Themes: ${themes.slice(0, 3).join(', ')}` : 'Themes: limited in the selection',
    freqMentions.length ? `Frequently mentioned: ${freqMentions.join(', ')}` : 'Frequently mentioned: none detected'
  ];
  const pointsCount = Math.min(Math.max(options.outputControls?.keyPointsCount || 3, 1), basePoints.length);
  const keyPoints = basePoints.slice(0, pointsCount);

  return { summary, keyPoints };
}

/**
 * Build strict-format prompt for LLM providers.
 * @param {Comment[]} comments
 * @param {FilterOption} filter
 */
function buildLLMPrompt(comments, filter) {
  const lines = comments.map(c => `- [${c.sentiment}] ${c.text}`);
  return [
    'You are an assistant summarizing e-consult user comments. Follow the rules strictly:',
    '1) Use only the provided comments. 2) Keep language professional and simple. 3) Output EXACTLY this structure:',
    'SUMMARY:\n<4-6 sentences overall summary>\n',
    'KEY POINTS:\n* point 1\n* point 2\n* point 3\n',
    'SENTIMENT OVERVIEW:\nPositive comments: <count>\nNegative comments: <count>\nNeutral comments: <count>\n',
    `Selection: ${filter}\nComments:\n${lines.join('\n')}`
  ].join('\n');
}

function isValidLLMResponse(text) {
  if (!text || typeof text !== 'string') return false;
  const upper = text.toUpperCase();
  return upper.includes('SUMMARY:') && upper.includes('KEY POINTS:') && upper.includes('SENTIMENT OVERVIEW:');
}

/**
 * Try LLM provider and fallback to heuristic if unavailable or response malformed.
 * @param {Comment[]} comments
 * @param {FilterOption} filter
 * @param {SummarizerOptions} options
 */
async function generateCommentSummary(comments, filter, options = {}) {
  const { filtered, counts } = filterComments(comments, filter);
  const chunkThreshold = options.chunkThreshold || 800;
  const chunkSize = options.chunkSize || 80;
  const doMapReduce = options.mapReduce ?? (filtered.length > chunkThreshold);

  async function summarizeSet(set) {
    const setCounts = countSentiments(set);
    if (options.provider && typeof options.provider.summarize === 'function') {
      try {
        const prompt = buildLLMPrompt(set, filter);
        const llmText = await options.provider.summarize(prompt);
        const text = typeof llmText === 'string' ? llmText.trim() : String(llmText);
        if (isValidLLMResponse(text)) {
          return text;
        }
      } catch (_) {
        // Fall through to heuristic.
      }
    }

    const h = buildHeuristicSummary(set, setCounts, filter, options);
    return [
      'SUMMARY:',
      h.summary,
      '',
      'KEY POINTS:',
      ...h.keyPoints.map(p => `* ${p}`),
      '',
      'SENTIMENT OVERVIEW:',
      `Positive comments: ${setCounts.positive}`,
      `Negative comments: ${setCounts.negative}`,
      `Neutral comments: ${setCounts.neutral}`,
    ].join('\n');
  }

  if (!doMapReduce) {
    const text = await summarizeSet(filtered);
    return { text, counts };
  }

  const chunks = chunkArray(filtered, chunkSize);
  const chunkSummaries = [];
  for (const ch of chunks) {
    const t = await summarizeSet(ch);
    chunkSummaries.push({ text: t, sentiment: 'neutral' });
  }
  const aggregateText = await summarizeSet(chunkSummaries.map(cs => ({ text: cs.text, sentiment: 'neutral' })));
  return { text: aggregateText, counts };
}

/**
 * Create an interactive session over a fixed filtered set.
 * @param {Comment[]} comments
 * @param {FilterOption} filter
 * @param {SummarizerOptions} options
 */
async function createInteractiveSummarizerSession(comments, filter, options = {}) {
  const { filtered, counts } = filterComments(comments, filter);
  const base = await generateCommentSummary(comments, filter, options);
  const maxExamples = options.maxExamplesPerPoint || 3;

  function examplesForKeywords(keywords, sentiment) {
    const subset = sentiment ? filtered.filter(c => c.sentiment === sentiment) : filtered;
    const out = [];
    for (const c of subset) {
      const textNorm = normalizeText(c.text);
      if (keywords.some(k => textNorm.includes(k))) {
        out.push({ sentiment: c.sentiment, text: c.text });
        if (out.length >= maxExamples) break;
      }
    }
    return out;
  }

  const overallStats = buildStats(filtered);
  const negStats = buildStats(filtered.filter(c => c.sentiment === 'negative'));
  const posStats = buildStats(filtered.filter(c => c.sentiment === 'positive'));

  return {
    summaryText: base.text,
    counts,
    explainPoint: (index) => {
      const points = [
        negStats.topWords.slice(0, 3),
        overallStats.topWords.slice(0, 3),
        posStats.topWords.slice(0, 3)
      ];
      const keywords = points[index - 1] || [];
      return {
        keywords,
        examples: examplesForKeywords(keywords)
      };
    },
    showExamplesFromComments: (keywords, sentiment) => examplesForKeywords(keywords, sentiment),
    generateMoreDetailedSummary: () => {
      const themes = overallStats.topWords.slice(0, 10).join(', ');
      const bigrams = overallStats.topBigrams.slice(0, 5).join(', ');
      return [
        'Detailed Summary:',
        `Themes observed: ${themes}.`,
        `Common phrases: ${bigrams}.`,
        `Positive mentions: ${posStats.topWords.slice(0, 5).join(', ')}.`,
        `Negative mentions: ${negStats.topWords.slice(0, 5).join(', ')}.`
      ].join('\n');
    },
    comparePositiveVsNegative: () => {
      return [
        'Comparison:',
        `Positive count: ${counts.positive}; Negative count: ${counts.negative}.`,
        `Positive themes: ${posStats.topWords.slice(0, 5).join(', ')}.`,
        `Negative themes: ${negStats.topWords.slice(0, 5).join(', ')}.`
      ].join('\n');
    },
    ask: async (question) => {
      // Retrieve top relevant comments by keyword overlap
      const qTokens = tokenize(question);
      const scored = filtered.map((c, idx) => {
        const ct = tokenize(c.text);
        const overlap = qTokens.reduce((acc, qt) => acc + (ct.includes(qt) ? 1 : 0), 0);
        return { idx, score: overlap, comment: c };
      }).filter(r => r.score > 0).sort((a, b) => b.score - a.score).slice(0, 25);
      const subset = scored.map(s => s.comment);

      if (subset.length === 0) {
        // Fallback: use overall top themes to construct a concise answer with examples
        const st = overallStats;
        const examples = filtered.slice(0, Math.min(maxExamples, 5)).map(c => ({ sentiment: c.sentiment, text: c.text }));
        const answerHeuristic = st.topWords.length
          ? `Based on the available comments, key topics include ${st.topWords.slice(0, 5).join(', ')}. Users commonly reference ${st.topBigrams.slice(0, 2).join(' and ') || 'various aspects'}.`
          : 'Comments do not reveal clear topics for this question.';
        if (options.provider) {
          try {
            const lines = filtered.slice(0, 150).map(c => `- [${c.sentiment}] ${c.text}`).join('\n');
            const prompt = [
              'Answer the user question using ONLY the provided comments. Keep it concise (3-5 sentences).',
              `Question: ${question}`,
              'Summary context (from previous step):',
              base.text,
              'Comments:',
              lines
            ].join('\n');
            const llmText = await options.provider.summarize(prompt);
            const answer = typeof llmText === 'string' ? llmText.trim() : String(llmText);
            return { answer, examples };
          } catch (_) {
            // fall through
          }
        }
        return { answer: answerHeuristic, examples };
      }

      if (options.provider) {
        try {
          const lines = subset.map(c => `- [${c.sentiment}] ${c.text}`).join('\n');
          const prompt = [
            'Answer the user question using ONLY the provided comments. Keep it concise (3-5 sentences).',
            `Question: ${question}`,
            'Summary context (from previous step):',
            base.text,
            'Comments:',
            lines
          ].join('\n');
          const llmText = await options.provider.summarize(prompt);
          const answer = typeof llmText === 'string' ? llmText.trim() : String(llmText);
          return { answer, examples: subset.slice(0, maxExamples).map(c => ({ sentiment: c.sentiment, text: c.text })) };
        } catch (_) {
          // fall back to heuristic
        }
      }

      // Heuristic: produce a brief answer from top keywords and examples
      const st = buildStats(subset);
      const answer = `Key points related to the question: ${st.topWords.slice(0, 5).join(', ')}.`;
      return { answer, examples: subset.slice(0, maxExamples).map(c => ({ sentiment: c.sentiment, text: c.text })) };
    }
  };
}

module.exports = {
  filterComments,
  generateCommentSummary,
  createInteractiveSummarizerSession,
};