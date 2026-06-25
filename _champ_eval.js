"use strict";
// Rigorous fixed-spawn verification of the deployed v3 champion (league_robust_v3_best_105 -> v3champ JSON),
// 120 ep/opp across 4 seed-bases, reporting win / fire% / hits-dealt / ttk vs the 4 laika.
const path = require("path");
const { RicochetCore, policyForward, ACTION_TABLE } = require(path.join(process.cwd(), "game_core.js"));
global.window = global.window || {};
require(path.join(process.cwd(), "model_weights_v3champ.js"));
const POL = global.window.RICOCHET_POLICIES.v3champ;
const OPPS = ["laika", "easy_laika", "stationary", "laika-aggressive-pro"];
const MAX = 3000, SEEDBASES = [300000, 500000, 700000, 900000], PER = 30;   // 4 x 30 = 120 ep/opp

function evalOpp(opp) {
  let win = 0, n = 0, fireActs = 0, steps = 0, hitsSum = 0, ttkSum = 0, ttkN = 0;
  for (const sb of SEEDBASES) for (let e = 0; e < PER; e++) {
    const seed = (sb + e * 131) >>> 0;
    const c = new RicochetCore({ seed, arenaMode: "survival", scenario: "moba1v1duel", spawnPowerups: true,
      shellDecay: true, ruleset: "survival_v1", spawnMode: "fixed", maxSteps: MAX });
    c.reset(seed);
    let info = null, k = 0, done = false;
    while (k < MAX) {
      const a = policyForward(c.observe(0), POL);
      if (ACTION_TABLE[a] && ACTION_TABLE[a][2]) fireActs++;
      steps++;
      const o = c.step(a, opp, { dt: 1 / 30, repeat: 2 }); info = o.info; k++;
      if (o.done) { done = true; break; }
    }
    n++;
    if (done && info && info.loserId === 1) { win++; ttkSum += (info.elapsed || 0); ttkN++; }
    if (info) hitsSum += (info.hitsDealt || 0);
  }
  return { win: win / n, fire_pct: 100 * fireActs / steps, hits: hitsSum / n, ttk: ttkN ? ttkSum / ttkN : 0, n };
}

console.log("=== v3 champion @ FIXED spawn (survival_v1), 120 ep/opp, 4 seed-bases ===");
const wins = [];
for (const o of OPPS) {
  const r = evalOpp(o); wins.push(r.win);
  console.log(`${o.padEnd(22)} win=${r.win.toFixed(3)}  fire%=${r.fire_pct.toFixed(1)}  hits=${r.hits.toFixed(2)}  ttk=${r.ttk.toFixed(1)}s  (n=${r.n})`);
}
const mean = wins.reduce((a, b) => a + b, 0) / wins.length, min = Math.min(...wins);
console.log(`MEAN ${mean.toFixed(3)}  MIN ${min.toFixed(3)}  cleared ${wins.filter((w) => w > 0.5).length}/4 cells >0.5`);
