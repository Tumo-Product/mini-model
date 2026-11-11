// Reference copy of irish-music-ai/src/tabs/deck/sets.js.
// Meter-aware scoring and retrain-set membership (sets: dark jigs, slip jigs).
// goodness() credits 6-beat bars; goodnessMeter() credits bars that fit the tune's own meter
// (6/8 -> 6 eighths, 9/8 -> 9), so a slip jig is judged as a slip jig. Everything else is identical.
import { bodyOf, keyOf, meterOf, goodness } from './jiglogic.mjs';

const NOTE_RE = /[_^=]*([A-Ga-gz])[,']*(\d*)(\/*)(\d*)/g;
function barDurations(body) {
  const clean = body.replace(/"[^"]*"|\{[^}]*\}|![^!]*!/g, '').replace(/\n/g, ' ');
  const durs = [];
  for (const chunk of clean.split(/\|+|:+/)) {
    let d = 0;
    NOTE_RE.lastIndex = 0;
    let m;
    while ((m = NOTE_RE.exec(chunk))) {
      let v = m[2] ? parseFloat(m[2]) : 1;
      if (m[3]) v /= m[4] ? parseFloat(m[4]) : 2 ** m[3].length;
      d += v;
    }
    if (d > 0) durs.push(d);
  }
  return durs;
}

export function eighthsPerBar(abc) {
  const m = meterOf(abc).match(/^(\d+)\s*\/\s*(\d+)$/);
  return m ? (Number(m[1]) * 8) / Number(m[2]) : 6;
}

export function meterFit(abc) {
  const want = eighthsPerBar(abc);
  const durs = barDurations(bodyOf(abc));
  const inner = durs.length > 2 ? durs.slice(1, -1) : durs;
  return inner.length ? inner.filter((d) => Math.abs(d - want) < 0.01).length / inner.length : 0;
}

export function goodnessMeter(abc, pool, copiedOverride = null) {
  const g = goodness(abc, pool, copiedOverride);
  const fit = meterFit(abc) >= 0.75 ? 1 : 0;
  const nBars = (bodyOf(abc).match(/\|/g) || []).length;
  const target = eighthsPerBar(abc) === 9 ? 16 : 18;
  const lenOk = nBars ? Math.max(0, 1 - Math.abs(Math.min(nBars, 40) - target) / target) : 0;
  const score = g.original * (0.30 * g.style + 0.25 * g.cadence + 0.15 * fit + 0.12 * g.smooth + 0.10 * lenOk + 0.08 * g.rep);
  return { ...g, score, fit, lenOk };
}

// Is one generated tune in a retrain set? dark: a minor or dorian key; slip: a 9/8 slip jig whose
// bars mostly hold 9 eighths. setCount() is what the deck shows as "before n/of -> after n/of".
const DARK_MODE = /^(m|min|minor|dor|dorian|aeo|aeolian)$/i;
export function inSet(abc, set) {
  if (!abc) return false;
  if (set === 'dark') {
    const k = keyOf(abc).replace(/\s+/g, '').match(/^[A-G][#b]?(.*)$/);
    return Boolean(k && DARK_MODE.test(k[1]));
  }
  if (set === 'slip') return eighthsPerBar(abc) === 9 && meterFit(abc) >= 0.75;
  return false;
}

export function setCount(abcs, set) {
  const list = (abcs || []).filter(Boolean);
  return { n: list.filter((a) => inSet(a, set)).length, of: list.length };
}
