// Lightweight Hugging Face Inference API client (text-generation)
// Requires env: HF_API_KEY
// Uses node-fetch via dynamic import to work in CommonJS.

const DEFAULT_MODEL = 'meta-llama/Llama-3.1-8B-Instruct';
const fetch = (url, options) => import('node-fetch').then(({ default: f }) => f(url, options));

/**
 * @param {{ model?: string }} options
 */
function createHuggingFaceProvider(options = {}) {
  const apiKey = process.env.HF_API_KEY || '';
  const model = options.model || DEFAULT_MODEL;

  async function summarize(prompt) {
    if (!apiKey) {
      throw new Error('HF_API_KEY not set');
    }
    const url = `https://api-inference.huggingface.co/models/${encodeURIComponent(model)}`;
    const payload = {
      inputs: prompt,
      parameters: {
        max_new_tokens: 350,
        temperature: 0.3,
        return_full_text: false
      }
    };

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
      throw new Error(`HF API error ${res.status}: ${body}`);
    }
    const data = await res.json();
    // HF can return array or object depending on model
    if (Array.isArray(data) && data[0]?.generated_text) {
      return data[0].generated_text;
    }
    if (data?.generated_text) return data.generated_text;
    if (Array.isArray(data) && typeof data[0] === 'string') return data[0];
    // Fallback to stringifying
    return typeof data === 'string' ? data : JSON.stringify(data);
  }

  return { name: 'huggingface', summarize };
}

module.exports = { createHuggingFaceProvider };
