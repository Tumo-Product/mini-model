import { goodness, trimToTwoParts, plagiarism, jigRules, styleMatch } from './jiglogic.mjs';
import { goodnessMeter, inSet, setCount, meterFit } from './sets.mjs';
import { passesFloor } from './floor.mjs';
import { readFileSync } from 'node:fs';
const { tunes, pool } = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const out = tunes.map((t) => {
  const two = trimToTwoParts(t);
  const g = goodness(two, pool);
  const gm = goodnessMeter(two, pool);
  return { two, score: g.score, style: g.style, cadence: g.cadence, six: g.six, smooth: g.smooth, lenOk: g.lenOk, rep: g.rep, copied: plagiarism(t, pool).frac,
    meter_score: gm.score, fit: gm.fit, floor: passesFloor(t), dark: inSet(t, 'dark'), slip: inSet(t, 'slip'), meterFit: meterFit(t), sixOk: jigRules(t).sixOk };
});
console.log(JSON.stringify({ out, dark: setCount(tunes, 'dark'), slip: setCount(tunes, 'slip') }));
