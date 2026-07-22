#!/usr/bin/env node
/**
 * Direct BFL concurrency probe.
 * Fires N create-job requests in parallel (no RPM throttle, no Vercel, no browser).
 * Use this to verify the documented ~24 active-task limit.
 *
 *   node scripts/bfl-concurrency-probe.js
 *   node scripts/bfl-concurrency-probe.js --n 30
 *   node scripts/bfl-concurrency-probe.js --n 30 --image https://example.com/room.jpg
 */

const fs = require('fs');
const path = require('path');

function loadEnvLocal() {
  const envPath = path.join(__dirname, '..', '.env.local');
  if (!fs.existsSync(envPath)) return;
  for (const line of fs.readFileSync(envPath, 'utf8').split('\n')) {
    const m = line.match(/^([^#=]+)=(.*)$/);
    if (!m) continue;
    const key = m[1].trim();
    const val = m[2].trim().replace(/^["']|["']$/g, '');
    if (!process.env[key]) process.env[key] = val;
  }
}

function arg(name, fallback) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx === -1) return fallback;
  return process.argv[idx + 1] ?? fallback;
}

loadEnvLocal();

const API_KEY = process.env.BFL_API_KEY;
const N = parseInt(arg('n', '30'), 10);
const MODEL = arg('model', 'flux-2-klein-9b');
const IMAGE = arg('image', '');
const PROMPT = arg('prompt', 'Make the room brighter');
const BASE = 'https://api.bfl.ai/v1';

if (!API_KEY) {
  console.error('Missing BFL_API_KEY in env or .env.local');
  process.exit(1);
}

if (!IMAGE) {
  console.error('Pass a public image URL: --image https://...');
  process.exit(1);
}

async function createJob(i) {
  const started = Date.now();
  try {
    const resp = await fetch(`${BASE}/${MODEL}`, {
      method: 'POST',
      headers: {
        accept: 'application/json',
        'Content-Type': 'application/json',
        'x-key': API_KEY,
      },
      body: JSON.stringify({
        prompt: PROMPT,
        input_image: IMAGE,
        output_format: 'jpeg',
        safety_tolerance: 2,
      }),
    });

    const text = await resp.text();
    let json;
    try { json = JSON.parse(text); } catch { json = { raw: text }; }

    return {
      i,
      status: resp.status,
      ok: resp.ok,
      ms: Date.now() - started,
      id: json.id || null,
      error: resp.ok ? null : text.slice(0, 300),
    };
  } catch (e) {
    return {
      i,
      status: 0,
      ok: false,
      ms: Date.now() - started,
      id: null,
      error: e.message,
    };
  }
}

(async () => {
  console.log(`Probing BFL with ${N} parallel creates → ${BASE}/${MODEL}`);
  console.log('(This bypasses the web UI RPM/concurrency model.)\n');

  const results = await Promise.all(
    Array.from({ length: N }, (_, i) => createJob(i + 1))
  );

  const ok = results.filter((r) => r.ok);
  const fail = results.filter((r) => !r.ok);
  const byStatus = {};
  for (const r of results) {
    byStatus[r.status] = (byStatus[r.status] || 0) + 1;
  }

  console.log('Status counts:', byStatus);
  console.log(`Success creates: ${ok.length}/${N}`);
  console.log(`Failed creates:  ${fail.length}/${N}`);

  if (fail.length) {
    console.log('\nSample failures:');
    for (const r of fail.slice(0, 5)) {
      console.log(`  #${r.i} HTTP ${r.status}: ${r.error}`);
    }
  }

  if (fail.some((r) => r.status === 429)) {
    console.log('\n✓ Got HTTP 429 — active-task limit is enforced on this key.');
  } else if (ok.length >= 25) {
    console.log('\n⚠ 25+ creates succeeded with no 429.');
    console.log('  This key/account likely allows more than the public default of 24,');
    console.log('  or BFL accepted them before rejecting later (check dashboard / poll).');
  } else {
    console.log('\nNo 429 observed. Try a larger --n (e.g. 40) or confirm image URL works.');
  }

  // Cancel/ignore leftover jobs — they will finish or expire on BFL side.
  console.log(`\nCreated job ids (first 5): ${ok.slice(0, 5).map((r) => r.id).join(', ') || '(none)'}`);
})();
