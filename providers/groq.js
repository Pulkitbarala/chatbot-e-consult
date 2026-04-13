// Groq chat completion provider.
// Requires env: GROQ_API_KEY
// Optional env: GROQ_MODEL

const DEFAULT_MODEL = 'llama3-8b-8192';
const fetch = (url, options) => import('node-fetch').then(({ default: f }) => f(url, options));

/**
 * @param {{ model?: string }} options
 */
function createGroqProvider(options = {}) {
  const apiKey = process.env.GROQ_API_KEY || '';
  const model = options.model || process.env.GROQ_MODEL || DEFAULT_MODEL;

  async function chat(messages, overrides = {}) {
    if (!apiKey) {
      throw new Error('GROQ_API_KEY not set');
    }
    if (!model) {
      throw new Error('GROQ_MODEL not set');
    }

    const payload = {
      model,
      messages,
      temperature: Number.isFinite(overrides.temperature) ? overrides.temperature : 0.3,
      max_tokens: Number.isFinite(overrides.max_tokens) ? overrides.max_tokens : 700,
    };

    const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`Groq error ${res.status}: ${body}`);
    }
    const data = await res.json();
    const content = data?.choices?.[0]?.message?.content;
    return typeof content === 'string' ? content.trim() : JSON.stringify(data);
  }

  return { name: 'groq', chat };
}

module.exports = { createGroqProvider };
