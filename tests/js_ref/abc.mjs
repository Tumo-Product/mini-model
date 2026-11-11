// Reference copy of irish-music-ai/src/tabs/deck/abc.js (player stubbed).
// Model output -> the ABC a card shows and plays. Body notes are never edited.
// Good/real tunes: the first full part (A-part) played once, through the
// ending that lands. Junk: the model's raw text as written, cut at one of its
// own barlines after about 8 s of sound.
const visualObjFromAbc = () => null; const msPerMeasureFor = () => 0;

const VALID_KEY = /^[A-G][b#]?(major|minor|dorian|mixolydian|aeolian|ionian|lydian|phrygian|locrian|maj|min|m|dor|mix)?$/i;
export const JUNK_CAP_MS = 8000;

// generations often run on into a second tune: keep the first one only
export function firstTune(raw) {
  const i = raw.indexOf('\nX:', 2);
  return i > 0 ? raw.slice(0, i) : raw;
}

export function parseGenerated(raw) {
  const first = firstTune(raw);
  const keyLines = [...first.matchAll(/^K:([^\n]*)\n?/gm)];
  const keys = keyLines.map((m) => m[1].trim()).filter((k) => VALID_KEY.test(k));
  const last = keyLines[keyLines.length - 1];
  const body = last ? first.slice(last.index + last[0].length) : first;
  return { key: keys.length ? keys[keys.length - 1] : null, body, readable: keyLines.length > 0, first };
}

// the tune's own meter (a slip jig stays 9/8); 6/8 when it gives none
const meterOf = (first) => { const m = /^M:\s*(\d+\/\d+)\s*$/m.exec(first || ''); return m ? m[1] : '6/8'; };
const header = (key, meter = '6/8') => `X:1\nM:${meter}\nL:1/8\nK:${key || 'C'}\n`;
const PART_END = /:\||\|\]|\|\|/;

// First part, played once. With 1st/2nd endings: everything before |1 plus
// the 2nd ending (the one that lands home).
export function aPart(body) {
  const flat = body
    .replace(/\r/g, '')
    .replace(/^\s*([A-Za-z]):([^\n]*)$/gm, '[$1:$2]')
    .replace(/\n\s*/g, ' ')
    .trim();
  const m = PART_END.exec(flat);
  let part;
  if (m) {
    const seg = flat.slice(0, m.index);
    const e1 = seg.search(/\|\s*\[?1|\[1/);
    if (e1 >= 0 && m[0] === ':|') {
      const rest = flat.slice(m.index + 2);
      const e2 = /^\s*\[?2\s*/.exec(rest);
      if (e2) {
        const after = rest.slice(e2[0].length);
        const stop = after.search(/\|\||\|\]|\|:|:\|/);
        const ending2 = (stop >= 0 ? after.slice(0, stop) : after).replace(/\|\s*$/, '');
        part = `${seg.slice(0, e1)}| ${ending2}`;
      } else {
        part = seg.slice(0, e1);
      }
    } else {
      part = seg;
    }
  } else {
    part = flat.split('|').filter((b) => b.trim()).slice(0, 8).join('|');
  }
  part = part.replace(/\|:/g, '|').replace(/::/g, '|').replace(/^\s*\|\s*/, '').replace(/\|\s*$/, '');
  return `${part} |]`;
}

// Where the first part ends in the flattened body, so the next part can follow it.
function firstPartEnd(flat) {
  const m = PART_END.exec(flat);
  if (!m) return -1;
  const seg = flat.slice(0, m.index);
  if (m[0] === ':|' && seg.search(/\|\s*\[?1|\[1/) >= 0) {
    const rest = flat.slice(m.index + 2);
    const e2 = /^\s*\[?2\s*/.exec(rest);
    if (e2) {
      const after = rest.slice(e2[0].length);
      const stop = after.search(/\|\||\|\]|\|:|:\|/);
      return stop >= 0 ? m.index + 2 + e2[0].length + stop : flat.length;
    }
  }
  return m.index + m[0].length;
}

const flatten = (body) => body
  .replace(/\r/g, '')
  .replace(/^\s*([A-Za-z]):([^\n]*)$/gm, '[$1:$2]')
  .replace(/\n\s*/g, ' ')
  .trim();

// The last sounding note's letter, to tell whether a part lands on the key's home note.
function lastNoteLetter(part) {
  const notes = part.replace(/"[^"]*"/g, '').replace(/![^!]*!/g, '').replace(/\[[A-Za-z]:[^\]]*\]/g, '').match(/[A-Ga-g](?=[,']*[0-9/]*[^A-Ga-g]*$)/g);
  return notes ? notes[notes.length - 1].toUpperCase() : null;
}

export const landsHome = (part, key) => Boolean(key) && lastNoteLetter(part) === key.trim()[0].toUpperCase();

// A model tune's first part, and when that part does not land on the home note,
// the next part too (each played once), so the card ends where the model resolved it.
// Notes are never edited; a tune whose parts never come home plays its first part only.
export function homePart(body, key) {
  const a = aPart(body);
  if (!key || landsHome(a.replace(/\|\]\s*$/, ''), key)) return { part: a, parts: 1 };
  const flat = flatten(body);
  const end = firstPartEnd(flat);
  if (end < 0 || end >= flat.length) return { part: a, parts: 1 };
  const restBody = flat.slice(end).replace(/^\s*\|?\]?/, '');
  if (!/[A-Ga-g]/.test(restBody)) return { part: a, parts: 1 };
  const b = aPart(restBody);
  if (!landsHome(b.replace(/\|\]\s*$/, ''), key)) return { part: a, parts: 1 };
  return { part: `${a.replace(/\|\]\s*$/, '')}| ${b}`, parts: 2 };
}

export function realPartAbc(abc) {
  const m = /^K:[^\n]*\n/m.exec(abc);
  const key = m ? m[0].slice(2).trim() : 'C';
  const body = m ? abc.slice(m.index + m[0].length) : abc;
  return header(key) + aPart(body);
}

// Junk: the model's exact text, no key or metre (none was written).
// Field-looking lines are made inert so the reader does not take them as headers.
export function junkAbc(raw) {
  const body = raw.replace(/^X:/, '').replace(/^([A-Za-z]):/gm, '$1 ').replace(/[%\\]/g, ' ');
  return `X:1\nL:1/8\nK:C\n${body}`;
}

function engrave(abc) {
  try { return visualObjFromAbc(abc); } catch { return null; }
}

function readsOk(abc) {
  const vo = engrave(abc);
  if (!vo) return false;
  try { vo.setTiming(120, 0); vo.setUpAudio({}); return true; } catch { return false; }
}

// Some soup crashes the music reader. Blank out (same length, so character
// positions still map onto the raw text) the fewest lines needed.
export function safeJunkAbc(raw) {
  let abc = junkAbc(raw);
  if (readsOk(abc)) return { abc, blanked: 0 };
  const head = abc.indexOf('K:C\n') + 4;
  const lines = abc.slice(head).split('\n');
  let blanked = 0;
  for (let i = 0; i < lines.length && !readsOk(abc); i++) {
    const t = abc.slice(0, head) + lines.map((l, j) => (j === i ? ' '.repeat(l.length) : l)).join('\n');
    if (readsOk(t)) { abc = t; blanked++; break; }
  }
  for (let i = lines.length - 1; i >= 0 && !readsOk(abc); i--) {
    lines[i] = ' '.repeat(lines[i].length);
    abc = abc.slice(0, head) + lines.join('\n');
    blanked++;
  }
  return { abc, blanked };
}

// Heard length of an ABC string in ms, from the same note schedule the synth
// renders (setUpAudio) and the same tempo rule the player uses.
export function heardMs(abc) {
  const vo = engrave(abc);
  if (!vo) return null;
  try {
    const mpm = msPerMeasureFor(vo, abc);
    const meter = vo.getMeterFraction ? vo.getMeterFraction() : { num: 1, den: 1 };
    const msPerWhole = mpm / ((meter.num / meter.den) || 1);
    const audio = vo.setUpAudio({});
    let end = 0;
    (audio.tracks || []).forEach((tr) => tr.forEach((n) => {
      if (n.cmd === 'note') end = Math.max(end, (n.start + n.duration) * msPerWhole);
    }));
    return end;
  } catch {
    return null;
  }
}

function bodyStart(abc) {
  const m = /^K:[^\n]*\n/m.exec(abc);
  return m ? m.index + m[0].length : 0;
}

// End offsets of every barline token in the body (outside quotes/comments).
function barlineEnds(abc, from) {
  const ends = [];
  let i = from;
  let q = false;
  while (i < abc.length) {
    const c = abc[i];
    if (c === '"') { q = !q; i++; continue; }
    if (!q && c === '|') {
      let j = i + 1;
      while (j < abc.length && /[|:\]]/.test(abc[j])) j++;
      ends.push(j);
      i = j;
      continue;
    }
    if (c === '\n') q = false;
    i++;
  }
  return ends;
}

// Cut at one of the model's own barlines so about `capMs` of sound remains.
// Never edits a note: the ABC is only shortened. Returns null when the whole
// text is already short enough.
export function capAtBarline(abc, capMs = JUNK_CAP_MS) {
  const total = heardMs(abc);
  if (total != null && total <= capMs * 1.15) return null;
  const ends = barlineEnds(abc, bodyStart(abc)).filter((e) => e < abc.length);
  const at = (e) => heardMs(abc.slice(0, e));
  if (ends.length) {
    let lo = 0;
    let hi = ends.length - 1;
    let best = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const ms = at(ends[mid]);
      if (ms != null && ms <= capMs) { best = mid; lo = mid + 1; } else hi = mid - 1;
    }
    const cands = [best, best + 1].filter((k) => k >= 0 && k < ends.length)
      .map((k) => ({ end: ends[k], ms: at(ends[k]) }))
      .filter((c) => c.ms != null && c.ms > 0 && c.ms <= capMs * 1.25);
    cands.sort((a, b) => Math.abs(a.ms - capMs) - Math.abs(b.ms - capMs));
    if (cands.length) {
      const c = cands[0];
      return { abc: abc.slice(0, c.end), ms: c.ms, totalMs: total, by: 'barline' };
    }
  }
  // no usable barline: cut at the model's own line break instead
  const nl = [];
  for (let i = bodyStart(abc); i < abc.length; i++) if (abc[i] === '\n') nl.push(i);
  const fits = nl.map((e) => ({ end: e, ms: at(e) })).filter((c) => c.ms != null && c.ms > 0 && c.ms <= capMs * 1.25);
  if (fits.length) {
    const c = fits[fits.length - 1];
    return { abc: abc.slice(0, c.end), ms: c.ms, totalMs: total, by: 'line' };
  }
  return null;
}

// -> { kind, abc, raw, rawShown, keyGuessed, blanked, cap }
export function cardFromGenerated(raw, { junkCap = false } = {}) {
  const g = parseGenerated(raw);
  if (!g.readable) {
    const j = safeJunkAbc(g.first);
    const cap = capAtBarline(j.abc);
    const abc = cap ? cap.abc : j.abc;
    const head = abc.indexOf('K:C\n') + 4;
    const shownLen = abc.length - head;
    return {
      kind: 'junk', abc, raw: g.first, rawShown: g.first.slice(0, shownLen + (g.first.startsWith('X:') ? 2 : 0)),
      blanked: j.blanked, cap: cap ? { ms: Math.round(cap.ms), totalMs: Math.round(cap.totalMs || 0), by: cap.by } : null,
    };
  }
  const hp = homePart(g.body, g.key);
  let abc = header(g.key, meterOf(g.first)) + hp.part;
  let cap = null;
  if (junkCap) {
    const c = capAtBarline(abc);
    if (c) { abc = c.abc; cap = { ms: Math.round(c.ms), totalMs: Math.round(c.totalMs || 0), by: c.by }; }
  }
  return { kind: junkCap ? 'rough' : 'tune', abc, raw: g.first, keyGuessed: !g.key, blanked: 0, cap, parts: hp.parts, home: landsHome(hp.part.replace(/\|\]\s*$/, ''), g.key) };
}
