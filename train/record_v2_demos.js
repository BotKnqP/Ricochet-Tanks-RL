"use strict";
// Record FRESH survival_v2 expert demos: the EXPERT script (default laika-aggressive, the v2 universal counter)
// plays blue (player 0) vs a pool of opponents under survival_v2/half_random, and we log obs->action transitions
// in the train_bc JSONL format (transition rows + an episode_summary row per episode). Only the obs/action of the
// learner-seat (player 0) is recorded; opponent identity is NOT in the obs. Keeps the v2 state distribution so a
// BC/DAgger seed isn't stale v1 data.
//   node train/record_v2_demos.js --expert laika-aggressive --opps laika,easy_laika,stationary,laika-aggressive,charger,laika-aggressive-pro --episodes 80 --out data/expert_demos/v2/aggro_v2.jsonl
const path = require("path"); const fs = require("fs");
const { RicochetCore, controlToAction, ACTION_TABLE } = require(path.join(process.cwd(), "game_core.js"));
function arg(n, d) { const i = process.argv.indexOf("--" + n); return i >= 0 ? process.argv[i + 1] : d; }
const EXPERT = arg("expert", "laika-aggressive");
const OPPS = arg("opps", "laika,easy_laika,stationary,laika-aggressive,charger,laika-aggressive-pro").split(",");
const EP_PER = parseInt(arg("episodes", "80"), 10);
const SPAWN = arg("spawn", "half_random");
const RULESET = arg("ruleset", "survival_v2");   // survival_v1 + half_random = v1.5 (v1 numbers, random spawn)
const SEEDBASE = parseInt(arg("seed", "770000"), 10);
const MAX = parseInt(arg("max-steps", "3000"), 10);
const OUT = arg("out", "data/expert_demos/v2/aggro_v2.jsonl");
const outAbs = path.isAbsolute(OUT) ? OUT : path.join(process.cwd(), OUT);
fs.mkdirSync(path.dirname(outAbs), { recursive: true });

// Target mode: --total-wins N --min-wins M records round-robin UNTIL total wins >= N AND each opponent has >= M
// wins (a satisfied opponent is skipped). Falls back to the fixed --episodes EP_PER mode when --total-wins is 0.
const TOTAL_WINS = parseInt(arg("total-wins", "0"), 10);
const MIN_WINS = parseInt(arg("min-wins", "0"), 10);
const fh = fs.openSync(outAbs, "w");
let ep = 0, kept = 0, wins = 0, mismatch = 0;
const winsByOpp = {}; for (const o of OPPS) winsByOpp[o] = 0;

function recordOne(opp) {                                           // record ONE episode for opp; write demo if a win
  const seed = (SEEDBASE + ep * 97) >>> 0; ep++;
  const c = new RicochetCore({ seed, arenaMode: "survival", scenario: "moba1v1duel", spawnPowerups: true,
    shellDecay: true, ruleset: RULESET, spawnMode: SPAWN, maxSteps: MAX });
  c.reset(seed);
  const rows = []; let info = null, k = 0;
  while (k < MAX) {
    const obs = c.observe(0);
    const b = c.scriptedControl(0, EXPERT);
    const throttle = b.throttle | 0, turn = b.turn | 0, fire = b.fire ? 1 : 0;
    const action = controlToAction(throttle, turn, fire);
    const tbl = ACTION_TABLE[action];                              // canonical control for this action
    if (!tbl || tbl[0] !== throttle || tbl[1] !== turn || Boolean(tbl[2]) !== Boolean(fire)) mismatch++;
    rows.push({ type: "transition", episode: ep, step: k, scenario: "moba1v1duel", opponent: opp,
      expert: EXPERT, obs: Array.from(obs, x => Math.round(x * 1e6) / 1e6),
      action, control: { throttle, turn, fire: !!fire } });
    const o = c.step(action, opp, { dt: 1 / 30, repeat: 2 }); info = o.info; k++;
    if (o.done) break;
  }
  const result = info && info.result === "win" ? "win" : (info && info.result === "loss" ? "loss" : "other");
  if (result === "win") {                                          // only keep WINS as demos (good_wins filter)
    for (const r of rows) fs.writeSync(fh, JSON.stringify(r) + "\n");
    fs.writeSync(fh, JSON.stringify({ type: "episode_summary", episode: ep, result, good_demo: true,
      opponent: opp, scenario: "moba1v1duel" }) + "\n");
    kept += rows.length; wins++; winsByOpp[opp]++;
  }
}

if (TOTAL_WINS > 0) {
  const satisfied = (o) => winsByOpp[o] >= MIN_WINS && wins >= TOTAL_WINS;
  const done = () => wins >= TOTAL_WINS && OPPS.every((o) => winsByOpp[o] >= MIN_WINS);
  let guard = 0;
  while (!done() && guard < 500000) {
    for (const opp of OPPS) { if (!satisfied(opp)) { recordOne(opp); guard++; } }
    process.stderr.write(`  ...${wins} wins ${JSON.stringify(winsByOpp)}\r`);
  }
} else {
  for (const opp of OPPS) for (let e = 0; e < EP_PER; e++) recordOne(opp);
}
fs.closeSync(fh);
console.log(`wrote ${outAbs}: ${ep} episodes, ${wins} wins kept (${kept} transitions), control_mismatch=${mismatch}`);
console.log("  wins by opponent:", JSON.stringify(winsByOpp));
