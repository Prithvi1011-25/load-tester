module.exports = (req, res) => {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const providers = {
    ai_studio: !!process.env.GEMINI_API_KEY,
    vertex_ai: !!(process.env.VERTEX_SERVICE_ACCOUNT_JSON && process.env.VERTEX_PROJECT),
    ai_gateway: !!process.env.AI_GATEWAY_API_KEY,
    openrouter: !!process.env.OPENROUTER_API_KEY,
    replicate: !!process.env.REPLICATE_API_TOKEN,
    bfl: !!process.env.BFL_API_KEY,
  };

  res.setHeader('Cache-Control', 'no-store');
  res.json({ providers });
};
