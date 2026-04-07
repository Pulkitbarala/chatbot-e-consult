// Hugging Face Inference API embeddings provider.
// Requires env: HF_API_KEY

const DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2';
const fetch = (url, options) => import('node-fetch').then(({ default: f }) => f(url, options));

/**
 * @param {{ model?: string }} options
 */
function createHfEmbeddingsProvider(options = {}) {
  const apiKey = process.env.HF_API_KEY || '';
  const model = options.model || DEFAULT_MODEL;

  async function embed(texts) {
    if (!apiKey) {
      throw new Error('HF_API_KEY not set');
    }
    if (!Array.isArray(texts)) {
      throw new Error('texts must be an array');
    }
    if (texts.length === 0) return [];

    const url = `https://api-inference.huggingface.co/models/${encodeURIComponent(model)}`;
    const payload = { inputs: texts };

    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`HF embeddings error ${res.status}: ${body}`);
    }

    const data = await res.json();
    if (!Array.isArray(data)) {
      throw new Error('Unexpected embeddings response');
    }
    return data.map(vec => Array.isArray(vec) ? vec.map(Number) : []);
  }

  return { name: 'hf-embeddings', embed };
}

module.exports = { createHfEmbeddingsProvider };
