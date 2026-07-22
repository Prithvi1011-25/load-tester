const { GoogleGenAI } = require('@google/genai');

let _aiStudioClient = null;
let _vertexClient = null;

function getAIStudioClient() {
  if (!_aiStudioClient) {
    _aiStudioClient = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  }
  return _aiStudioClient;
}

function getVertexClient() {
  if (!_vertexClient) {
    const credentials = JSON.parse(process.env.VERTEX_SERVICE_ACCOUNT_JSON);
    if (credentials.private_key) {
      credentials.private_key = credentials.private_key.replace(/\\n/g, '\n');
    }
    _vertexClient = new GoogleGenAI({
      vertexai: true,
      project: process.env.VERTEX_PROJECT,
      location: process.env.VERTEX_LOCATION || 'global',
      googleAuthOptions: {
        credentials,
        projectId: process.env.VERTEX_PROJECT,
        scopes: ['https://www.googleapis.com/auth/cloud-platform'],
      },
    });
  }
  return _vertexClient;
}

function classifyError(error) {
  const msg = (error.message || '').toLowerCase();
  const status = error.status || error.statusCode || error.httpStatusCode;

  if (status === 429 || msg.includes('rate limit') || msg.includes('quota') || msg.includes('resource_exhausted'))
    return 'rate_limit';
  if (status === 402 || msg.includes('credit'))
    return 'rate_limit';
  if (status === 408 || msg.includes('timeout') || msg.includes('etimedout') || msg.includes('deadline') || msg.includes('aborted'))
    return 'timeout';
  if (status === 401 || status === 403 || msg.includes('api key') || msg.includes('unauthorized'))
    return 'auth_error';
  if (msg.includes('safety') || msg.includes('blocked') || msg.includes('moderated'))
    return 'safety_block';
  if (status === 400 || msg.includes('invalid') || msg.includes('requires input_image'))
    return 'invalid_request';
  if (status >= 500 || msg.includes('internal') || msg.includes('unavailable'))
    return 'server_error';
  if (msg.includes('econnrefused') || msg.includes('enotfound') || msg.includes('network'))
    return 'network';
  return 'unknown';
}

// ─── Google (AI Studio / Vertex) ────────────────────────────────────────────────

async function runGeminiRequest(client, { model, prompt, images, maxTokens, temperature }) {
  const parts = [];

  if (images && images.length > 0) {
    for (const img of images) {
      parts.push({ inlineData: { mimeType: img.mimeType, data: img.base64 } });
    }
  }

  parts.push({ text: prompt });

  const isImageModel = model.includes('image');

  const config = {
    temperature: temperature ?? 1.0,
  };
  if (maxTokens) config.maxOutputTokens = maxTokens;

  if (isImageModel) {
    config.responseModalities = ['TEXT', 'IMAGE'];
  }

  const response = await client.models.generateContent({
    model,
    contents: [{ role: 'user', parts }],
    config,
  });

  let responseChars = 0;
  let hasImage = false;
  let responseText = '';

  if (response.candidates && response.candidates[0]) {
    const candidate = response.candidates[0];
    if (candidate.content && candidate.content.parts) {
      for (const part of candidate.content.parts) {
        if (part.text) {
          responseChars += part.text.length;
          responseText += part.text;
        }
        if (part.inlineData) hasImage = true;
      }
    }
  } else if (response.text) {
    responseChars = response.text.length;
    responseText = response.text;
  }

  return { responseChars, hasImage, responseText: responseText.slice(0, 500) };
}

// ─── Vercel AI Gateway ──────────────────────────────────────────────────────────

async function runAIGatewayRequest({ model, prompt, images, maxTokens, temperature }) {
  const contentParts = [];

  if (images && images.length > 0) {
    for (const img of images) {
      contentParts.push({
        type: 'image_url',
        image_url: { url: `data:${img.mimeType};base64,${img.base64}` },
      });
    }
  }

  contentParts.push({ type: 'text', text: prompt });

  const body = {
    model,
    messages: [{ role: 'user', content: contentParts }],
    ...(maxTokens ? { max_tokens: maxTokens } : {}),
    temperature: temperature ?? 1.0,
    stream: false,
  };

  const resp = await fetch('https://ai-gateway.vercel.sh/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${process.env.AI_GATEWAY_API_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const errBody = await resp.text();
    const err = new Error(`AI Gateway ${resp.status}: ${errBody}`);
    err.status = resp.status;
    throw err;
  }

  const data = await resp.json();
  const text = data.choices?.[0]?.message?.content || '';
  return { responseChars: text.length, hasImage: false, responseText: text.slice(0, 500) };
}

// ─── OpenRouter ─────────────────────────────────────────────────────────────────

async function runOpenRouterRequest({ model, prompt, images, maxTokens, temperature }) {
  const contentParts = [];

  if (images && images.length > 0) {
    for (const img of images) {
      contentParts.push({
        type: 'image_url',
        image_url: { url: `data:${img.mimeType};base64,${img.base64}` },
      });
    }
  }

  contentParts.push({ type: 'text', text: prompt });

  const body = {
    model,
    messages: [{ role: 'user', content: contentParts }],
    ...(maxTokens ? { max_tokens: maxTokens } : {}),
    temperature: temperature ?? 1.0,
    stream: false,
  };

  const resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const errBody = await resp.text();
    const err = new Error(`OpenRouter ${resp.status}: ${errBody}`);
    err.status = resp.status;
    throw err;
  }

  const data = await resp.json();
  const text = data.choices?.[0]?.message?.content || '';
  return { responseChars: text.length, hasImage: false, responseText: text.slice(0, 500) };
}

// ─── Black Forest Labs (BFL) ────────────────────────────────────────────────────
// Aligned with prod: POST /v1/{model} → poll polling_url or GET /v1/get_result?id=

const BFL_BASE_URL = 'https://api.bfl.ai/v1';
const BFL_POLL_INTERVAL_MS = 2000;
const BFL_MAX_POLL_ATTEMPTS = 60; // ~120s
const BFL_READY_STATUSES = new Set(['ready', 'completed']);
const BFL_TERMINAL_FAIL = new Set([
  'failed',
  'error',
  'request moderated',
  'content moderated',
  'moderated',
]);

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function extractBFLOutput(result) {
  const r = result?.result || result || {};
  return r.sample || r.image_url || r.output || '';
}

async function runBFLRequest({ model, prompt, images, imageUrl, outputFormat, safetyTolerance }) {
  const apiKey = process.env.BFL_API_KEY;
  const endpoint = String(model || '').replace(/^\//, '') || 'flux-2-klein-9b';

  const body = {
    prompt,
    output_format: outputFormat || 'jpeg',
    safety_tolerance: safetyTolerance ?? 2,
  };

  const publicUrl = typeof imageUrl === 'string' ? imageUrl.trim() : '';
  if (publicUrl) {
    body.input_image = publicUrl;
  } else if (images && images.length > 0) {
    body.input_image = images[0].base64;
    if (images[1]) body.input_image_2 = images[1].base64;
    if (images[2]) body.input_image_3 = images[2].base64;
  } else {
    const err = new Error('BFL requires input_image (public URL preferred, or uploaded base64)');
    err.status = 400;
    throw err;
  }

  const controller = new AbortController();
  const createTimeout = setTimeout(() => controller.abort(), 120000);

  let submitResp;
  try {
    submitResp = await fetch(`${BFL_BASE_URL}/${endpoint}`, {
      method: 'POST',
      headers: {
        accept: 'application/json',
        'Content-Type': 'application/json',
        'x-key': apiKey,
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (e) {
    if (e.name === 'AbortError') {
      const err = new Error('BFL create timeout (120s)');
      err.status = 408;
      throw err;
    }
    throw e;
  } finally {
    clearTimeout(createTimeout);
  }

  if (!submitResp.ok) {
    const errBody = await submitResp.text();
    const err = new Error(`BFL ${submitResp.status}: ${errBody}`);
    err.status = submitResp.status;
    throw err;
  }

  const submitData = await submitResp.json();
  const jobId = submitData.id;
  const pollingUrl = submitData.polling_url
    || (jobId ? `${BFL_BASE_URL}/get_result?id=${encodeURIComponent(jobId)}` : null);

  if (!pollingUrl) {
    throw new Error(`BFL response missing id/polling_url: ${JSON.stringify(submitData)}`);
  }

  for (let attempt = 0; attempt < BFL_MAX_POLL_ATTEMPTS; attempt++) {
    await sleep(BFL_POLL_INTERVAL_MS);

    const pollResp = await fetch(pollingUrl, {
      headers: {
        accept: 'application/json',
        'x-key': apiKey,
      },
    });

    if (!pollResp.ok) {
      const errBody = await pollResp.text();
      const err = new Error(`BFL poll ${pollResp.status}: ${errBody}`);
      err.status = pollResp.status;
      throw err;
    }

    const result = await pollResp.json();
    const status = String(result.status || '').toLowerCase();

    if (BFL_READY_STATUSES.has(status)) {
      const output = extractBFLOutput(result);
      const outputStr = typeof output === 'string' ? output : JSON.stringify(output);
      return {
        responseChars: outputStr.length,
        hasImage: !!outputStr,
        responseText: outputStr.slice(0, 500),
      };
    }

    if (BFL_TERMINAL_FAIL.has(status)) {
      throw new Error(`BFL generation ${result.status}: ${JSON.stringify(result)}`);
    }
  }

  const timeoutErr = new Error('BFL timeout: result not ready within ~120s (60 poll attempts)');
  throw timeoutErr;
}

// ─── Replicate ──────────────────────────────────────────────────────────────────

async function runReplicateRequest({ model, prompt, images, maxTokens }) {
  const input = { prompt };

  if (images && images.length > 0) {
    input.image = `data:${images[0].mimeType};base64,${images[0].base64}`;
    if (images[1]) input.image_2 = `data:${images[1].mimeType};base64,${images[1].base64}`;
    if (images[2]) input.image_3 = `data:${images[2].mimeType};base64,${images[2].base64}`;
  }

  if (maxTokens) input.max_new_tokens = maxTokens;

  const resp = await fetch(`https://api.replicate.com/v1/models/${model}/predictions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${process.env.REPLICATE_API_TOKEN}`,
      Prefer: 'wait',
    },
    body: JSON.stringify({ input }),
  });

  if (!resp.ok) {
    const errBody = await resp.text();
    const err = new Error(`Replicate ${resp.status}: ${errBody}`);
    err.status = resp.status;
    throw err;
  }

  const data = await resp.json();

  if (data.status === 'failed') {
    throw new Error(`Replicate prediction failed: ${data.error || 'unknown'}`);
  }

  const output = data.output;
  const outputStr = typeof output === 'string' ? output : JSON.stringify(output);
  const responseChars = outputStr.length;
  return { responseChars, hasImage: Array.isArray(output) && typeof output[0] === 'string' && output[0].startsWith('http'), responseText: outputStr.slice(0, 500) };
}

// ─── Handler ────────────────────────────────────────────────────────────────────

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const {
    provider,
    model,
    prompt,
    images,
    imageUrl,
    maxTokens,
    temperature,
    outputFormat,
    safetyTolerance,
  } = req.body;

  if (!provider || !model || !prompt) {
    return res.status(400).json({ error: 'Missing required fields: provider, model, prompt' });
  }

  const start = Date.now();

  try {
    let result;

    switch (provider) {
      case 'ai_studio': {
        const client = getAIStudioClient();
        result = await runGeminiRequest(client, { model, prompt, images, maxTokens, temperature });
        break;
      }
      case 'vertex_ai': {
        const client = getVertexClient();
        result = await runGeminiRequest(client, { model, prompt, images, maxTokens, temperature });
        break;
      }
      case 'ai_gateway': {
        result = await runAIGatewayRequest({ model, prompt, images, maxTokens, temperature });
        break;
      }
      case 'openrouter': {
        result = await runOpenRouterRequest({ model, prompt, images, maxTokens, temperature });
        break;
      }
      case 'replicate': {
        result = await runReplicateRequest({ model, prompt, images, maxTokens });
        break;
      }
      case 'bfl': {
        result = await runBFLRequest({
          model,
          prompt,
          images,
          imageUrl,
          outputFormat,
          safetyTolerance,
        });
        break;
      }
      default:
        return res.status(400).json({ error: `Unknown provider: ${provider}` });
    }

    const latencyMs = Date.now() - start;

    res.json({
      success: true,
      latencyMs,
      responseChars: result.responseChars,
      hasImage: result.hasImage || false,
      responseText: result.responseText || '',
    });
  } catch (error) {
    const latencyMs = Date.now() - start;

    res.json({
      success: false,
      latencyMs,
      errorCategory: classifyError(error),
      errorMessage: error.message || 'Unknown error',
    });
  }
};
