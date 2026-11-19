// Reference copy of irish-music-ai/src/jigLogic.jsx (pure functions only, React parts removed).
// Used by tests/test_score_port.py to check the Python port. Copied from irish-music-ai on 2026-11-11; re-copy if the activity scoring changes.

// ---- Shared jig logic + result-card pieces, used by both workspaces --------
// (the from-scratch WorkspaceTab and the cached/finetune WorkspaceFastTab).

export function bodyOf(abc) { const m = abc.match(/^K:[^\n]*\n/m); return m ? abc.slice(abc.indexOf(m[0]) + m[0].length) : abc; }
export function keyOf(abc)  { const m = abc.match(/^K:([^\n]*)/m); return m ? m[1].trim() : 'D'; }
export function meterOf(abc){ const m = abc.match(/^M:([^\n]*)/m); return m ? m[1].trim() : '6/8'; }

const NOTE_RE = /[_^=]*([A-Ga-gz])[,']*(\d*)(\/*)(\d*)/g;
function barDurations(body) {
  const clean = body.replace(/"[^"]*"|\{[^}]*\}|![^!]*!/g, '').replace(/\n/g, ' ');
  const durs = [];
  for (const chunk of clean.split(/\|+|:+/)) {
    let d = 0; NOTE_RE.lastIndex = 0; let m;
    while ((m = NOTE_RE.exec(chunk))) {
      let v = m[2] ? parseFloat(m[2]) : 1;
      if (m[3]) v = v / (m[4] ? parseFloat(m[4]) : Math.pow(2, m[3].length));
      d += v;
    }
    if (d > 0) durs.push(d);
  }
  return durs;
}

// the two jig-rule checks measured from the output
export function jigRules(abc) {
  const body = bodyOf(abc);
  const durs = barDurations(body);
  const inner = durs.length > 2 ? durs.slice(1, -1) : durs;
  const six = inner.length ? inner.filter(d => Math.abs(d - 6) < 0.01).length / inner.length : 0;
  const parts = [];
  for (const seg of body.replace(/\n/g, ' ').split(/:\||\|\||\|:|::/)) {
    const bars = seg.split('|').filter(b => /[A-Ga-g]/.test(b)).length;
    if (bars >= 4) parts.push(bars);
  }
  const eightBar = parts.length >= 2 && parts.filter(p => p === 8 || p === 9 || p === 16).length / parts.length >= 0.5;
  return { sixOk: six >= 0.75, eightBar };
}

// longest run copied verbatim from the jigs it trained on
// Longest verbatim run the tune shares with any pool tune, as a fraction of the
// tune. Binary-search on match length (was O(pool·len²) and froze the page when
// called 3× on every dial click); memoized because cached tunes never change.
const _plagCache = new Map();
export function plagiarism(abc, pool) {
  const g = bodyOf(abc);
  const n = g.length;
  if (!n || !pool || !pool.length) return { frac: 0, span: '', title: null };
  const key = pool.length + '|' + g;
  const hit = _plagCache.get(key);
  if (hit) return hit;
  const bodies = pool.map(t => ({ title: t.title, body: bodyOf(t.abc) }));
  const poolText = bodies.map(b => b.body).join('');
  const M = poolText.length;
  const findAt = (L) => {                     // some g-substring of length L present in poolText?
    if (L <= 0 || L > n) return null;
    for (let i = 0; i + L <= n; i++) { const s = g.substr(i, L); if (poolText.indexOf(s) >= 0) return s; }
    return null;
  };
  let lo = 1, hi = n, bestSub = '';
  while (lo <= hi) { const mid = (lo + hi) >> 1; const s = findAt(mid); if (s) { bestSub = s; lo = mid + 1; } else hi = mid - 1; }
  let title = null;
  if (bestSub) { const b = bodies.find(b => b.body.includes(bestSub)); title = b ? b.title : null; }
  const res = { frac: bestSub.length / n, span: bestSub, title };
  _plagCache.set(key, res);
  return res;
}

export const amountWord = (n, max) => { const r = n / max; return r < 0.15 ? 'a tiny set' : r < 0.4 ? 'a small set' : r < 0.85 ? 'most of them' : 'every jig'; };
export const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// how different the 3 tunes are from each other (0 = clones, 1 = all distinct).
// cheap 3-gram Jaccard distance averaged over pairs — display-only (RL wiring later).
function _trigrams(s) { const set = new Set(); for (let i = 0; i + 3 <= s.length; i++) set.add(s.substr(i, 3)); return set; }
function _jaccardDiff(a, b) {
  const A = _trigrams(a), B = _trigrams(b);
  if (!A.size && !B.size) return 0;
  let inter = 0; for (const x of A) if (B.has(x)) inter++;
  const uni = A.size + B.size - inter;
  return uni ? 1 - inter / uni : 0;
}
export function varietyOf(cards) {
  const bodies = (cards || []).map(c => bodyOf(c.abc).replace(/\s+/g, ''));
  if (bodies.length < 2) return null;
  let pairs = 0, sum = 0;
  for (let i = 0; i < bodies.length; i++) for (let j = i + 1; j < bodies.length; j++) { pairs++; sum += _jaccardDiff(bodies[i], bodies[j]); }
  return pairs ? sum / pairs : null;
}

// ---- 1-bit gauge widgets ----------------------------------------------------

// labelled ✓/✗ tick box: filled black square = pass, outline square = fail
// 1-bit hover tooltip (styles in index.html). Use for explaining a term.
// ---- trim a generation down to ONE clean tune (the model sometimes runs on:
// extra parts or a whole second tune, up to the 500-char cap) -----------------
export function trimToOneTune(abc) {
  if (!abc) return abc;
  const second = abc.indexOf('X:', 3);            // a second tune header?
  let t = second > 3 ? abc.slice(0, second).replace(/\s+$/, '') : abc;
  const kEnd = t.indexOf('\n', t.lastIndexOf('K:'));
  if (kEnd > 0) {
    const head = t.slice(0, kEnd + 1);
    let body = t.slice(kEnd + 1);
    const MAX = 300;                              // keep it to a readable single tune
    if (body.length > MAX) {
      let cut = body.lastIndexOf('|', MAX);       // end on a barline
      if (cut < 40) cut = MAX;
      body = body.slice(0, cut + 1);
    }
    t = head + body;
  }
  return t;
}

// ---- "sounds like a real jig": melodic-phrase similarity to the corpus -------
// Fraction of the tune's 8-note melodic shapes (as intervals, so key-independent)
// that ALSO occur in genuine jigs — WITHOUT being a verbatim copy (that's plagiarism).
const _DIA = { C: 0, D: 1, E: 2, F: 3, G: 4, A: 5, B: 6 };
function pitchSeq(abc) {
  const body = bodyOf(abc).replace(/"[^"]*"|\{[^}]*\}|![^!]*!/g, '');
  const re = /[_^=]*([A-Ga-gz])([,']*)/g; let m; const seq = [];
  while ((m = re.exec(body))) {
    const l = m[1]; if (l === 'z' || l === 'Z') continue;
    const lower = l >= 'a' && l <= 'g';
    const up = (m[2].match(/'/g) || []).length, dn = (m[2].match(/,/g) || []).length;
    seq.push(_DIA[l.toUpperCase()] + (lower ? 7 : 0) + 7 * up - 7 * dn);
  }
  return seq;
}
const _KG = 8;
function intervalGrams(abc) {
  const s = pitchSeq(abc); const iv = [];
  for (let i = 1; i < s.length; i++) iv.push(s[i] - s[i - 1]);
  const grams = [];
  for (let i = 0; i + _KG <= iv.length; i++) grams.push(iv.slice(i, i + _KG).join(','));
  return { grams, ivLen: iv.length };
}
let _stylePool = null, _stylePoolKey = null;
function stylePoolSet(pool) {
  if (_stylePool && _stylePoolKey === pool.length) return _stylePool;
  const set = new Set();
  for (const t of pool) for (const g of intervalGrams(t.abc).grams) set.add(g);
  _stylePool = set; _stylePoolKey = pool.length; return set;
}
export function styleMatch(abc, pool) {
  if (!pool || !pool.length) return 0;
  const { grams, ivLen } = intervalGrams(abc);
  if (ivLen < _KG) return 0;
  const set = stylePoolSet(pool);
  const covered = new Array(ivLen).fill(false);
  for (let i = 0; i < grams.length; i++) if (set.has(grams[i])) for (let j = i; j < i + _KG; j++) covered[j] = true;
  return covered.filter(Boolean).length / ivLen;
}
// ---- "what makes a jig GOOD" — measured against the 482 real jigs -----------
// The style gauge alone misses what listeners actually feel: real jigs RESOLVE
// (parts land on the tonic), stay a tidy two parts (not a run-on), move smoothly
// (few big leaps), and repeat their own motifs. `goodness` folds those in.

// split a body into its parts (segments ending in an end-repeat / double bar)
function splitParts(body) {
  const parts = []; let cur = '';
  for (const tk of body.split(/(:\||\|\|)/)) { cur += tk; if (tk === ':|' || tk === '||') { parts.push(cur); cur = ''; } }
  if (cur.trim()) parts.push(cur);
  return parts.filter(p => (p.match(/[A-Ga-g]/g) || []).length > 3);
}

// Trim a generation to a clean TWO-part jig: drop extra run-on parts / a second
// tune, strip chord symbols (random on a generated tune → clashy accompaniment),
// and make sure it ends on a repeat so it resolves instead of being chopped.
export function trimToTwoParts(abc) {
  if (!abc) return abc;
  const body = bodyOf(abc);
  const header = abc.slice(0, abc.indexOf(body));
  let b = body; const second = b.indexOf('X:'); if (second > 0) b = b.slice(0, second);
  b = b.replace(/"[^"]*"/g, '');                 // drop chord symbols
  const parts = splitParts(b);
  let kept = parts.length ? parts.slice(0, 2).join('') : b;
  kept = kept.replace(/\s+$/, '');
  if (!/(:\||\|\|?)$/.test(kept)) kept += ' :|';  // land on a cadence
  return (header + kept).replace(/\s+$/, '');
}

// parts should end on the tonic (or dominant) — the "it resolves" feeling
function cadenceScore(abc) {
  const tonic = _DIA[(keyOf(abc)[0] || 'D').toUpperCase()] ?? 1;
  const parts = splitParts(bodyOf(abc));
  if (!parts.length) return 0;
  let hits = 0;
  for (const p of parts) { const s = pitchSeq(p); if (!s.length) continue; const last = ((s[s.length - 1] % 7) + 7) % 7; if (last === tonic || last === (tonic + 4) % 7) hits++; }
  return hits / parts.length;
}

// how much the tune repeats its own 4-note motifs (catchiness)
function selfRepScore(abc) {
  const s = pitchSeq(abc); const win = [];
  for (let i = 0; i + 4 <= s.length; i++) win.push(s.slice(i, i + 4).join(','));
  if (!win.length) return 0; const seen = new Set(); let rep = 0;
  for (const w of win) { if (seen.has(w)) rep++; else seen.add(w); }
  return rep / win.length;
}

// A single 0..1 "good jig" score + its component parts, for ranking generations
// and for a richer gauge. A memorised copy (copied >= 20%) scores 0 — it may
// sound great, but it is not a good ORIGINAL tune.
export function goodness(abc, pool, copiedOverride = null) {
  const style = styleMatch(abc, pool);
  const cadence = cadenceScore(abc);
  const six = jigRules(abc).sixOk ? 1 : 0;
  const s = pitchSeq(abc); const iv = [];
  for (let i = 1; i < s.length; i++) iv.push(Math.abs(s[i] - s[i - 1]));
  const bigLeap = iv.length ? iv.filter(a => a >= 5).length / iv.length : 1;
  const smooth = Math.max(0, Math.min(1, 1 - bigLeap / 0.10));
  const nBars = (bodyOf(abc).match(/\|/g) || []).length;
  const lenOk = nBars ? Math.max(0, 1 - Math.abs(Math.min(nBars, 40) - 18) / 18) : 0;
  const rep = selfRepScore(abc);
  // copied is best measured on the first WHOLE generated tune, not the 2-part
  // trim (trimming shrinks the denominator and over-flags good tunes); callers
  // pass it in. Hard-zero only at true memorisation (>=35%) — below that it's
  // incidental idiomatic overlap, not plagiarism.
  const copied = copiedOverride != null ? copiedOverride : plagiarism(abc, pool).frac;
  const original = copied >= 0.35 ? 0 : 1;
  const score = original * (0.30 * style + 0.25 * cadence + 0.15 * six + 0.12 * smooth + 0.10 * lenOk + 0.08 * rep);
  return { score, style, cadence, six, smooth, lenOk, rep, copied, original };
}

// ---- note-level repairs, run on trimToTwoParts output -----------------------
// After trimming, the one genuine musical defect is that parts often don't
// RESOLVE. forceCadence rewrites an unresolved part's final note to the
// nearest-register tonic — a single in-register note edit that never adds a
// leap — and fixBraces drops orphan { }. Ornaments (~ rolls, {graces}) and
// durations are preserved. (Validated: +28% goodness, cadence to ceiling.)
const _headOf = (abc) => abc.slice(0, abc.indexOf(bodyOf(abc)));
const _maskGraces = (body) => body.replace(/\{[^}]*\}/g, (m) => ' '.repeat(m.length));
const _CORE_RE = /([_^=]*)([A-Ga-g])([,']*)/g;
function _repairTokens(masked) {
  const out = []; let m; _CORE_RE.lastIndex = 0;
  while ((m = _CORE_RE.exec(masked))) {
    const letter = m[2], oct = m[3] || '';
    const up = (oct.match(/'/g) || []).length, dn = (oct.match(/,/g) || []).length;
    const lower = letter >= 'a' && letter <= 'g';
    const deg = _DIA[letter.toUpperCase()] + (lower ? 7 : 0) + 7 * up - 7 * dn;
    out.push({ start: m.index, end: m.index + m[0].length, deg, letter });
  }
  return out;
}
function _renderDeg(D, letter) {
  const base = _DIA[letter.toUpperCase()]; const steps = Math.round((D - base) / 7);
  if (steps >= 1) return letter.toLowerCase() + "'".repeat(steps - 1);
  if (steps === 0) return letter.toUpperCase();
  return letter.toUpperCase() + ','.repeat(-steps);
}
function _nearestTonic(neighbourDeg, tonicLetterDeg) {
  let best = null, bd = 1e9;
  for (let st = -2; st <= 2; st++) { const D = tonicLetterDeg + 7 * st; const d = Math.abs(D - neighbourDeg); if (d < bd) { bd = d; best = D; } }
  return best;
}
export function forceCadence(abc, { onlyIfUnresolved = true } = {}) {
  const key = keyOf(abc); const tonic = _DIA[(key[0] || 'D').toUpperCase()] ?? 1; const dom = (tonic + 4) % 7; const letter = (key[0] || 'D').toUpperCase();
  const head = _headOf(abc); let body = bodyOf(abc);
  const bs = []; { const re = /:\||\|\|/g, masked = _maskGraces(body); let m; while ((m = re.exec(masked))) bs.push(m.index); }
  bs.push(body.length);
  const segs = []; let prev = 0; for (const b of bs) { segs.push([prev, b]); prev = b; }
  for (let i = segs.length - 1; i >= 0; i--) {
    const [s, e] = segs[i];
    const toks = _repairTokens(_maskGraces(body.slice(s, e)));
    if (toks.length < 4) continue;
    const last = toks[toks.length - 1]; const lastPc = ((last.deg % 7) + 7) % 7;
    if (onlyIfUnresolved && (lastPc === tonic || lastPc === dom)) continue;
    const neighbour = toks.length > 1 ? toks[toks.length - 2].deg : last.deg;
    const newCore = _renderDeg(_nearestTonic(neighbour, tonic), letter);
    body = body.slice(0, s + last.start) + newCore + body.slice(s + last.end);
  }
  return head + body;
}
export function fixBraces(abc) {
  const head = _headOf(abc); let body = bodyOf(abc);
  const op = (body.match(/\{/g) || []).length, cl = (body.match(/\}/g) || []).length;
  if (op === cl) return abc;
  let out = '', depth = 0;
  for (const ch of body) {
    if (ch === '{') { depth++; out += ch; }
    else if (ch === '}') { if (depth > 0) { depth--; out += ch; } }
    else out += ch;
  }
  if (depth > 0) out = out.replace(/\{([^{}]*)$/, '$1');
  return head + out;
}
export function repairJig(abc) { return forceCadence(fixBraces(abc), { onlyIfUnresolved: true }); }
