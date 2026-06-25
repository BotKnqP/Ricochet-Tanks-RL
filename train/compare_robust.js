"use strict";
// Robustness PROMOTION GATE (Phase E). Diffs two eval_metrics.js JSONs (baseline vs candidate) and applies
// the multi-criteria gate (GPT-aligned). The point is NOT "beat more scripts" but: improve robustness to
// UNSEEN behaviour without collapsing the known scripts or the safety metrics.
//   node train/compare_robust.js --base baseline.json --cand cand.json \
//        --fixed stationary,easy_laika,laika,laika-aggressive,laika-aggressive-pro \
//        --held precision,counter,turtle,baiter
// Everything not in --fixed/--held is treated as TRAIN-weird (randomized + trained presets).
const fs = require("fs");
function arg(n, d) { const i = process.argv.indexOf("--" + n); return i >= 0 ? process.argv[i + 1] : d; }
const base = JSON.parse(fs.readFileSync(arg("base"), "utf8"));
const cand = JSON.parse(fs.readFileSync(arg("cand"), "utf8"));
const FIXED = arg("fixed", "stationary,easy_laika,laika,laika-aggressive,laika-aggressive-pro").split(",");
// p-kiter is ALSO held-out: its 0.05 script-mix weight floored to 0 training envs (see allocate() rounding),
// so neither the anchor nor the v2 pilot ever trained on it -> a clean 5th generalization test.
const HELD = arg("held", "precision,counter,turtle,baiter,p-kiter").split(",");
const opps = Object.keys(cand).filter(o => base[o]);
const TRAIN = opps.filter(o => !FIXED.includes(o) && !HELD.includes(o));

function fmt(v) { return v == null ? "  -  " : (v >= 0 ? "+" : "") + v.toFixed(2); }
function row(o) {
  const b = base[o], c = cand[o];
  return `  ${o.padEnd(22)} win ${b.win.toFixed(2)}->${c.win.toFixed(2)} (${fmt(c.win - b.win)})  ` +
    `psnDth ${b.poison_death.toFixed(2)}->${c.poison_death.toFixed(2)}  selfHit ${b.self_hits.toFixed(2)}->${c.self_hits.toFixed(2)}`;
}
function meanWin(list) { return list.reduce((a, o) => a + cand[o].win - base[o].win, 0) / (list.length || 1); }
function meanOf(list, k, src) { return list.reduce((a, o) => a + (src[o][k] || 0), 0) / (list.length || 1); }

console.log("=== FIXED (must not collapse) ==="); FIXED.forEach(o => opps.includes(o) && console.log(row(o)));
console.log("=== TRAIN-weird (should rise) ==="); TRAIN.forEach(o => console.log(row(o)));
console.log("=== HELD-OUT (generalization: should rise, never trained/gated) ==="); HELD.forEach(o => opps.includes(o) && console.log(row(o)));

const fixedDrop = FIXED.filter(o => opps.includes(o)).map(o => base[o].win - cand[o].win);
const worstFixedDrop = Math.max(0, ...fixedDrop);
const trainDelta = meanWin(TRAIN), heldDelta = meanWin(HELD.filter(o => opps.includes(o)));
const psnBase = meanOf(opps, "poison_death", base), psnCand = meanOf(opps, "poison_death", cand);
const shBase = meanOf(opps, "self_hits", base), shCand = meanOf(opps, "self_hits", cand);
const vulnList = opps.filter(o => base[o].win_enemy_pup != null && cand[o].win_enemy_pup != null);
const vulnBase = meanOf(vulnList, "win_enemy_pup", base), vulnCand = meanOf(vulnList, "win_enemy_pup", cand);

const C = [
  ["fixed scripts don't collapse (worst drop <= 0.15)", worstFixedDrop <= 0.15, `worst drop ${worstFixedDrop.toFixed(2)}`],
  ["train-weird average improves", trainDelta > 0.0, `mean dWin ${fmt(trainDelta)}`],
  ["HELD-OUT average improves (true generalization)", heldDelta > 0.0, `mean dWin ${fmt(heldDelta)}`],
  ["poison_death down", psnCand <= psnBase + 0.02, `${psnBase.toFixed(2)}->${psnCand.toFixed(2)}`],
  ["powerup-vuln not worse (win when enemy got pup)", vulnCand >= vulnBase - 0.05, `${vulnBase.toFixed(2)}->${vulnCand.toFixed(2)}`],
  ["self-hits don't explode (<= 1.5x)", shCand <= shBase * 1.5 + 0.1, `${shBase.toFixed(2)}->${shCand.toFixed(2)}`],
];
console.log("\n=== PROMOTION GATE ===");
C.forEach(([name, ok, detail]) => console.log(`  [${ok ? "PASS" : "FAIL"}] ${name.padEnd(50)} ${detail}`));
const promote = C.every(c => c[1]);
console.log(`\nVERDICT: ${promote ? "PROMOTE" : "HOLD"}  (held-out dWin ${fmt(heldDelta)} is the headline generalization number)`);
