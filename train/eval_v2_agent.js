"use strict";
// Definitive v2 evaluation for a SINGLE identity-blind agent (no router). Evaluates a trained POLICY (exported
// json) OR a SCRIPT (e.g. laika-aggressive) as blue vs the full opponent suite under survival_v2, and reports
// per-opponent win + MAXIMIN + MEAN, split into a TRAIN family and a HELD-OUT family (opponents never trained on)
// so the generalization gap is explicit. Uses the same core.step cadence as the browser (dt 1/30, repeat 2).
//   node train/eval_v2_agent.js --policy runs/auto_live/v2agent.json --episodes 20 --spawn half_random
//   node train/eval_v2_agent.js --script laika-aggressive --episodes 20
const path = require("path"); const fs = require("fs");
const { RicochetCore, policyForward, controlToAction } = require(path.join(process.cwd(), "game_core.js"));
function arg(n, d) { const i = process.argv.indexOf("--" + n); return i >= 0 ? process.argv[i + 1] : d; }
const POLICY_PATH = arg("policy", "");
const SCRIPT = arg("script", "");
const POLICY = POLICY_PATH ? JSON.parse(fs.readFileSync(POLICY_PATH, "utf8")) : null;
const EPISODES = parseInt(arg("episodes", "20"), 10);
const SEEDBASES = arg("seeds", "300000,900000").split(",").map(Number);
const SPAWN = arg("spawn", "half_random");
const RULESET = arg("ruleset", "survival_v2");   // survival_v1 = the v1 multi-strategy regime (HP=3, fixed spawn)
const MAX = parseInt(arg("max-steps", "3000"), 10);
// TRAIN family = opponents the agent was DAgger'd against; HELD-OUT = styles it never saw (clean generalization).
const TRAIN = (arg("train", "laika,easy_laika,stationary,laika-aggressive,charger,laika-aggressive-pro")).split(",");
const HELD = (arg("held", "precision,counter,turtle,baiter,p-kiter")).split(",");

function actFor(c) {
  if (SCRIPT) { const b = c.scriptedControl(0, SCRIPT); return controlToAction(b.throttle, b.turn, b.fire); }
  return policyForward(c.observe(0), POLICY);
}
function evalOpp(opp) {
  let wins = 0, n = 0, ttkSum = 0, ttkN = 0, trunc = 0;
  for (const sb of SEEDBASES) for (let e = 0; e < EPISODES; e++) {
    const seed = (sb + e * 131) >>> 0;
    const c = new RicochetCore({ seed, arenaMode: "survival", scenario: "moba1v1duel", spawnPowerups: true,
      shellDecay: true, ruleset: RULESET, spawnMode: SPAWN, maxSteps: MAX });
    c.reset(seed);
    let info = null, k = 0, resolved = false;
    while (k < MAX) { const o = c.step(actFor(c), opp, { dt: 1 / 30, repeat: 2 }); info = o.info; k++; if (o.done) { resolved = true; break; } }
    n++;
    if (info && info.result === "win") { wins++; ttkSum += info.elapsed; ttkN++; }
    if (!resolved) trunc++;
  }
  return { win: wins / n, ttk: ttkN ? ttkSum / ttkN : null, trunc: trunc / n };
}

const who = SCRIPT ? `script:${SCRIPT}` : `policy:${path.basename(POLICY_PATH)}`;
console.log(`v2 AGENT EVAL  ${who}  (spawn=${SPAWN}, ${EPISODES}ep x ${SEEDBASES.length} seeds)`);
console.log("opponent".padEnd(22) + ["win", "ttk", "trunc"].map(h => h.padStart(8)).join("") + "   family");
const res = {};
function run(list, fam) { const ws = []; for (const opp of list) { const m = evalOpp(opp); res[opp] = { ...m, fam }; ws.push(m.win);
  console.log(opp.padEnd(22) + [m.win, m.ttk == null ? -1 : m.ttk, m.trunc].map(v => v.toFixed(2).padStart(8)).join("") + `   ${fam}`); } return ws; }
const trainW = run(TRAIN, "train");
console.log("-".repeat(48));
const heldW = run(HELD, "held-out");
console.log("=".repeat(48));
const all = [...trainW, ...heldW];
const mm = a => Math.min(...a).toFixed(2), mean = a => (a.reduce((x, y) => x + y, 0) / a.length).toFixed(2);
console.log(`TRAIN    maximin=${mm(trainW)} mean=${mean(trainW)}`);
console.log(`HELD-OUT maximin=${mm(heldW)} mean=${mean(heldW)}`);
console.log(`OVERALL  maximin=${mm(all)} mean=${mean(all)}`);
const dest = arg("out", ""); if (dest) fs.writeFileSync(dest, JSON.stringify(res, null, 2));
