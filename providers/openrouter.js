// OpenRouter chat completion provider.
// Requires env: OPENROUTER_API_KEY
// Optional env: OPENROUTER_MODEL, OPENROUTER_APP_URL, OPENROUTER_APP_NAME

const DEFAULT_MODEL = 'openrouter/auto';
const fetch = (url, options) => import('node-fetch').then(({ default: f }) => f(url, options));

/**
 * @param {{ model?: string }} options
 */
function createOpenRouterProvider(options = {}) {
  const apiKey = process.env.OPENROUTER_API_KEY || '';
  const model = options.model || process.env.OPENROUTER_MODEL || DEFAULT_MODEL;
  const appUrl = process.env.OPENROUTER_APP_URL || '';
  const appName = process.env.OPENROUTER_APP_NAME || 'ecs-summarizer';

  async function chat(messages, overrides = {}) {
    if (!apiKey) {
      throw new Error('OPENROUTER_API_KEY not set');
    }
    if (!model) {
      throw new Error('OPENROUTER_MODEL not set');
    }

    const payload = {
      model,
      messages,
      temperature: Number.isFinite(overrides.temperature) ? overrides.temperature : 0.3,
      max_tokens: Number.isFinite(overrides.max_tokens) ? overrides.max_tokens : 700,
    };

    const headers = {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    };
    if (appUrl) headers['HTTP-Referer'] = appUrl;
    if (appName) headers['X-Title'] = appName;

    const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`OpenRouter error ${res.status}: ${body}`);
    }
    const data = await res.json();
    const content = data?.choices?.[0]?.message?.content;
    return typeof content === 'string' ? content.trim() : JSON.stringify(data);
  }

  return { name: 'openrouter', chat };
}

module.exports = { createOpenRouterProvider };
