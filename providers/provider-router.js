// Provider router with hedged requests for chat completions.

const { createOpenRouterProvider } = require('./openrouter');
const { createGroqProvider } = require('./groq');
let createHuggingFaceProvider;
try {
  ({ createHuggingFaceProvider } = require('./huggingface'));
} catch (e) {
  createHuggingFaceProvider = null;
}

function toInt(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

const DEFAULT_TIMEOUT_MS = toInt(process.env.PROVIDER_TIMEOUT_MS, 15000);
const DEFAULT_HEDGE_DELAY_MS = toInt(process.env.PROVIDER_HEDGE_DELAY_MS, 900);

function messagesToPrompt(messages) {
  return messages
    .map(m => `${String(m.role || '').toUpperCase()}: ${String(m.content || '')}`)
    .join('\n\n');
}

async function callProvider(provider, messages, overrides) {
  if (provider.chat) {
    return provider.chat(messages, overrides);
  }
  if (provider.summarize) {
    const prompt = messagesToPrompt(messages);
    return provider.summarize(prompt);
  }
  throw new Error(`Provider ${provider.name || 'unknown'} does not support chat`);
}

function withTimeout(promise, timeoutMs) {
  if (!timeoutMs) return promise;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error('Provider timeout'));
    }, timeoutMs);
    promise
      .then((value) => {
        clearTimeout(timer);
        resolve(value);
      })
      .catch((err) => {
        clearTimeout(timer);
        reject(err);
      });
  });
}

function createProviderRouter(options = {}) {
  const providers = [];
  const hedgeDelayMs = toInt(options.hedgeDelayMs, DEFAULT_HEDGE_DELAY_MS);
  const timeoutMs = toInt(options.timeoutMs, DEFAULT_TIMEOUT_MS);

  if (options.openrouter !== false && process.env.OPENROUTER_API_KEY) {
    providers.push(createOpenRouterProvider({ model: options.openrouterModel }));
  }
  if (options.groq !== false && process.env.GROQ_API_KEY) {
    providers.push(createGroqProvider({ model: options.groqModel }));
  }
  if (options.huggingface !== false && process.env.HF_API_KEY && createHuggingFaceProvider) {
    providers.push(createHuggingFaceProvider({ model: options.hfModel || process.env.HF_CHAT_MODEL }));
  }

  if (providers.length === 0) {
    throw new Error('No chat providers configured');
  }

  async function chat(messages, overrides = {}) {
    let settled = false;
    const errors = [];

    return new Promise((resolve, reject) => {
      providers.forEach((provider, index) => {
        const delay = index === 0 ? 0 : hedgeDelayMs * index;
        setTimeout(() => {
          if (settled) return;
          const req = withTimeout(callProvider(provider, messages, overrides), timeoutMs);
          req
            .then((result) => {
              if (settled) return;
              settled = true;
              resolve(result);
            })
            .catch((err) => {
              errors.push({ provider: provider.name || 'unknown', error: err?.message || String(err) });
              if (errors.length >= providers.length && !settled) {
                settled = true;
                const message = errors.map(e => `${e.provider}: ${e.error}`).join('; ');
                reject(new Error(message));
              }
            });
        }, delay);
      });
    });
  }

  return { chat };
}

module.exports = { createProviderRouter };
