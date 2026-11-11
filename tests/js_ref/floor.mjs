// Reference copy of passesFloor from irish-music-ai/src/tabs/deck/data.js.
import { parseGenerated, homePart, landsHome } from './abc.mjs';
export const FLOOR_RULE = 'samples that read, land on their home note and run 6-24 bars are dealt first';
export const passesFloor = (raw) => {
  const g = parseGenerated(raw);
  if (!g.readable || !g.key) return false;
  const hp = homePart(g.body, g.key);
  const bars = hp.part.split('|').filter((b) => /[A-Ga-g]/.test(b)).length;
  return landsHome(hp.part.replace(/\|\]\s*$/, ''), g.key) && bars >= 6 && bars <= 24;
};
