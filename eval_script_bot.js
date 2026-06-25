"use strict";
// Script-vs-script evaluator for moba1v1duel (and other scenarios).
// Runs N episodes of blue-script vs red-script and prints combat metrics as JSON.
// Blue's scripted control is discretised through controlToAction() — identical to how
// the demo recorder/generator produce actions — so this measures exactly the policy a
// BC dataset would clone. No torch/PPO needed; pure game_core.js stepping.
//
//   node eval_script_bot.js --blue laika-aggressive-pro --red easy_laika --episodes 200
const { RicochetCore, controlToAction, actionToControl, ACTION_TABLE } = require("./game_core.js");

function arg(name, def) {
  const i = process.argv.indexOf("--" + name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}
const BLUE = arg("blue", "laika-aggressive-pro");
const RED = arg("red", "easy_laika");
const SCEN = arg("scenario", "moba1v1duel");
const EPISODES = parseInt(arg("episodes", "200"), 10);
const SEED = parseInt(arg("seed", "1307"), 10);
const MAX = parseInt(arg("max-steps", "1800"), 10);
const REC_DT = 1 / 30, REC_REPEAT = 2;   // training cadence

const counts = { win: 0, loss: 0, draw: 0, timeout: 0, other: 0 };
const rew = [], len = [], el = [], hd = [], ht = [], sh = [], pu = [], fh = [], oh = [];
let fire = 0, total = 0;
const actCounts = new Array(ACTION_TABLE.length).fill(0);

for (let ep = 0; ep < EPISODES; ep++) {
  const c = new RicochetCore({ seed: SEED + ep, arenaMode: "survival", scenario: SCEN, maxSteps: MAX, spawnPowerups: true });
  c.reset(SEED + ep);
  let steps = 0, totR = 0, info = null;
  while (!c.isDone()) {
    const bc = c.scriptedControl(0, BLUE);                       // blue = the expert under test
    const a = controlToAction(bc.throttle, bc.turn, bc.fire);    // -> Discrete(18) id (what BC would clone)
    const o = c.step(a, RED, { dt: REC_DT, repeat: REC_REPEAT });
    info = o.info; totR += o.reward; steps += 1;
    actCounts[a] += 1; total += 1;
    if (actionToControl(a).fire) fire += 1;
  }
  const res = info.result;
  counts[res in counts ? res : "other"] += 1;
  rew.push(totR); len.push(steps); el.push(info.elapsed);
  hd.push(info.hitsDealt || 0); ht.push(info.hitsTaken || 0); sh.push(info.selfHits || 0);
  pu.push(info.powerups || 0); fh.push(info.learnerHealth || 0); oh.push(info.opponentHealth || 0);
}

const n = Math.max(1, EPISODES);
const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const round = (v, d = 4) => Math.round(v * 10 ** d) / 10 ** d;
console.log(JSON.stringify({
  blue: BLUE, red: RED, scenario: SCEN, episodes: EPISODES,
  win_rate: round(counts.win / n), loss_rate: round(counts.loss / n),
  draw_rate: round(counts.draw / n), timeout_rate: round(counts.timeout / n),
  avg_reward: round(mean(rew)), avg_length: round(mean(len), 1), avg_elapsed: round(mean(el), 2),
  avg_hits_dealt: round(mean(hd), 3), avg_hits_taken: round(mean(ht), 3), avg_self_hits: round(mean(sh), 3),
  avg_final_health: round(mean(fh), 3), opponent_final_health: round(mean(oh), 3),
  powerups_collected: round(mean(pu), 3), fire_pct: round(100 * fire / Math.max(1, total), 2),
  action_distribution: actCounts,
}, null, 2));
